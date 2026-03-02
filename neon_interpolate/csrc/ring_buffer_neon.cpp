#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

constexpr int kPrecision = 12;

inline double aa_filter(double x) {
  x = std::abs(x);
  return (x < 1.0) ? 1.0 - x : 0.0;
}

inline double compute_scale(int64_t in_size, int64_t out_size) {
  return static_cast<double>(in_size) / static_cast<double>(out_size);
}

struct InterpolationWeights {
  std::vector<int64_t> xmin;
  std::vector<int64_t> xsize;
  std::vector<int16_t> weights;
  int kmax;
  unsigned int precision;
};

InterpolationWeights compute_weights(
    int64_t in_size,
    int64_t out_size,
    int64_t stride) {

  InterpolationWeights result;
  double scale = compute_scale(in_size, out_size);

  constexpr int interp_size = 2;
  double support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
  int max_interp_size = static_cast<int>(std::ceil(support) * 2 + 1);

  while (max_interp_size % 2 != 0) {
    max_interp_size += 1;
  }

  result.xmin.resize(out_size);
  result.xsize.resize(out_size);
  result.weights.resize(out_size * max_interp_size, 0);
  result.kmax = max_interp_size;

  double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;

  for (int64_t i = 0; i < out_size; i++) {
    double center = scale * (i + 0.5);

    int64_t xmin_val = std::max(
        static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
    int64_t xsize_val = std::min(
        static_cast<int64_t>(center + support + 0.5), in_size) - xmin_val;
    xsize_val = std::max(xsize_val, static_cast<int64_t>(0));
    xsize_val = std::min(xsize_val, static_cast<int64_t>(max_interp_size));

    std::vector<double> wt_f(max_interp_size, 0.0);
    double wt_sum = 0.0;
    for (int64_t j = 0; j < xsize_val; j++) {
      double w = aa_filter((j + xmin_val - center + 0.5) * invscale);
      wt_f[j] = w;
      wt_sum += w;
    }

    if (wt_sum != 0.0) {
      for (int64_t j = 0; j < xsize_val; j++) {
        wt_f[j] /= wt_sum;
      }
    }

    result.xmin[i] = xmin_val * stride;
    result.xsize[i] = xsize_val;

    for (int64_t j = 0; j < xsize_val; j++) {
      double v = wt_f[j] * (1 << kPrecision);
      result.weights[i * max_interp_size + j] = static_cast<int16_t>(v < 0 ? v - 0.5 : v + 0.5);
    }
    for (int64_t j = xsize_val; j < max_interp_size; j++) {
      result.weights[i * max_interp_size + j] = 0;
    }
  }

  result.precision = kPrecision;
  return result;
}

#if defined(__aarch64__)

inline int32x4_t neon_madd_s16(int16x8_t a, int16x8_t b) {
  int32x4_t prod_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
  int32x4_t prod_high = vmull_s16(vget_high_s16(a), vget_high_s16(b));
  return vpaddq_s32(prod_low, prod_high);
}

// Vertical conv reading from precomputed row pointers (ring buffer slots).
// Same NEON strategy as tiled_neon, but uses row_ptrs[i] + j instead of
// lineIn + ids_min + i * data_size.
void NeonRingVerticalConvolution8u(
    uint8_t* lineOut,
    const uint8_t* const* row_ptrs,
    int64_t xsize,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const int64_t data_size = xsize * num_channels;

  const int32_t initial_val = 1 << (coefs_precision - 1);
  const int32x4_t initial = vdupq_n_s32(initial_val);

  int64_t j = 0;

  for (; j + 16 <= data_size; j += 16) {
    int32x4_t sss0 = initial;
    int32x4_t sss1 = initial;
    int32x4_t sss2 = initial;
    int32x4_t sss3 = initial;

    int64_t i = 0;
    for (; i + 1 < ids_size; i += 2) {
      int16_t w0 = k[i];
      int16_t w1 = k[i + 1];
      int16x8_t mmk = {w0, w1, w0, w1, w0, w1, w0, w1};

      uint8x16_t src1 = vld1q_u8(row_ptrs[i] + j);
      uint8x16_t src2 = vld1q_u8(row_ptrs[i + 1] + j);

      uint8x16x2_t interleaved = vzipq_u8(src1, src2);

      uint8x8_t inter_lo_lo = vget_low_u8(interleaved.val[0]);
      uint8x8_t inter_lo_hi = vget_high_u8(interleaved.val[0]);
      uint8x8_t inter_hi_lo = vget_low_u8(interleaved.val[1]);
      uint8x8_t inter_hi_hi = vget_high_u8(interleaved.val[1]);

      int16x8_t pix0 = vreinterpretq_s16_u16(vmovl_u8(inter_lo_lo));
      int16x8_t pix1 = vreinterpretq_s16_u16(vmovl_u8(inter_lo_hi));
      int16x8_t pix2 = vreinterpretq_s16_u16(vmovl_u8(inter_hi_lo));
      int16x8_t pix3 = vreinterpretq_s16_u16(vmovl_u8(inter_hi_hi));

      sss0 = vaddq_s32(sss0, neon_madd_s16(pix0, mmk));
      sss1 = vaddq_s32(sss1, neon_madd_s16(pix1, mmk));
      sss2 = vaddq_s32(sss2, neon_madd_s16(pix2, mmk));
      sss3 = vaddq_s32(sss3, neon_madd_s16(pix3, mmk));
    }

    for (; i < ids_size; i++) {
      int16_t w = k[i];
      int16x8_t mmk = vdupq_n_s16(w);

      uint8x16_t src = vld1q_u8(row_ptrs[i] + j);

      uint8x8_t src_lo = vget_low_u8(src);
      uint8x8_t src_hi = vget_high_u8(src);

      int16x8_t pix_lo = vreinterpretq_s16_u16(vmovl_u8(src_lo));
      int16x8_t pix_hi = vreinterpretq_s16_u16(vmovl_u8(src_hi));

      int32x4_t prod0 = vmull_s16(vget_low_s16(pix_lo), vget_low_s16(mmk));
      int32x4_t prod1 = vmull_s16(vget_high_s16(pix_lo), vget_high_s16(mmk));
      int32x4_t prod2 = vmull_s16(vget_low_s16(pix_hi), vget_low_s16(mmk));
      int32x4_t prod3 = vmull_s16(vget_high_s16(pix_hi), vget_high_s16(mmk));

      sss0 = vaddq_s32(sss0, prod0);
      sss1 = vaddq_s32(sss1, prod1);
      sss2 = vaddq_s32(sss2, prod2);
      sss3 = vaddq_s32(sss3, prod3);
    }

    int32x4_t shift = vdupq_n_s32(-static_cast<int32_t>(coefs_precision));
    sss0 = vshlq_s32(sss0, shift);
    sss1 = vshlq_s32(sss1, shift);
    sss2 = vshlq_s32(sss2, shift);
    sss3 = vshlq_s32(sss3, shift);

    int16x4_t narrow0 = vqmovn_s32(sss0);
    int16x4_t narrow1 = vqmovn_s32(sss1);
    int16x4_t narrow2 = vqmovn_s32(sss2);
    int16x4_t narrow3 = vqmovn_s32(sss3);

    int16x8_t narrow_lo = vcombine_s16(narrow0, narrow1);
    int16x8_t narrow_hi = vcombine_s16(narrow2, narrow3);

    uint8x8_t result_lo = vqmovun_s16(narrow_lo);
    uint8x8_t result_hi = vqmovun_s16(narrow_hi);

    vst1_u8(lineOut + j, result_lo);
    vst1_u8(lineOut + j + 8, result_hi);
  }

  for (; j < data_size; j++) {
    int32_t sss = initial_val;

    for (int64_t i = 0; i < ids_size; i++) {
      sss += k[i] * static_cast<int32_t>(row_ptrs[i][j]);
    }

    sss >>= coefs_precision;
    sss = std::max(0, std::min(255, sss));
    lineOut[j] = static_cast<uint8_t>(sss);
  }
}

void NeonResampleHorizontalConvolution8u(
    uint8_t* lineOut,
    int64_t out_xsize,
    const uint8_t* lineIn,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const int64_t stride = num_channels;
  const int32_t initial_val = 1 << (coefs_precision - 1);

  for (int64_t out_x = 0; out_x < out_xsize; out_x++) {
    const int64_t ids_min = idx_ptr_xmin[out_x];
    const int64_t ids_size = idx_ptr_size[out_x];
    const int16_t* k = &kk[out_x * kmax];

    const uint8_t* lineIn_min = lineIn + ids_min;

    int32x4_t acc_r = vdupq_n_s32(0);
    int32x4_t acc_g = vdupq_n_s32(0);
    int32x4_t acc_b = vdupq_n_s32(0);

    int64_t i = 0;

    for (; i + 8 <= ids_size; i += 8) {
      uint8x8x3_t rgb = vld3_u8(lineIn_min + stride * i);
      int16x8_t weights = vld1q_s16(&k[i]);

      int16x8_t r16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[0]));
      int16x8_t g16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[1]));
      int16x8_t b16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[2]));

      acc_r = vmlal_s16(acc_r, vget_low_s16(r16), vget_low_s16(weights));
      acc_r = vmlal_s16(acc_r, vget_high_s16(r16), vget_high_s16(weights));
      acc_g = vmlal_s16(acc_g, vget_low_s16(g16), vget_low_s16(weights));
      acc_g = vmlal_s16(acc_g, vget_high_s16(g16), vget_high_s16(weights));
      acc_b = vmlal_s16(acc_b, vget_low_s16(b16), vget_low_s16(weights));
      acc_b = vmlal_s16(acc_b, vget_high_s16(b16), vget_high_s16(weights));
    }

    for (; i + 4 <= ids_size; i += 4) {
      uint8x8x3_t rgb = vld3_u8(lineIn_min + stride * i);
      int16x4_t weights4 = vld1_s16(&k[i]);

      int16x4_t r16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[0])));
      int16x4_t g16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[1])));
      int16x4_t b16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[2])));

      acc_r = vmlal_s16(acc_r, r16, weights4);
      acc_g = vmlal_s16(acc_g, g16, weights4);
      acc_b = vmlal_s16(acc_b, b16, weights4);
    }

    int32_t sum_r = vaddvq_s32(acc_r) + initial_val;
    int32_t sum_g = vaddvq_s32(acc_g) + initial_val;
    int32_t sum_b = vaddvq_s32(acc_b) + initial_val;

    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p = lineIn_min + stride * i;
      sum_r += w * p[0];
      sum_g += w * p[1];
      sum_b += w * p[2];
    }

    uint8_t* out = lineOut + stride * out_x;
    out[0] = static_cast<uint8_t>(std::clamp(sum_r >> coefs_precision, 0, 255));
    out[1] = static_cast<uint8_t>(std::clamp(sum_g >> coefs_precision, 0, 255));
    out[2] = static_cast<uint8_t>(std::clamp(sum_b >> coefs_precision, 0, 255));
  }
}

void ring_neon_bilinear_uint8(const at::Tensor& input, const at::Tensor& output) {
  TORCH_CHECK(input.size(0) == 1, "Only batch size 1 supported");
  TORCH_CHECK(input.size(1) == 3, "Only 3 channels (RGB) supported");
  TORCH_CHECK(input.is_contiguous(at::MemoryFormat::ChannelsLast),
              "Input must be channels_last");
  TORCH_CHECK(output.is_contiguous(at::MemoryFormat::ChannelsLast),
              "Output must be channels_last");

  int64_t C = input.size(1);
  int64_t H_in = input.size(2);
  int64_t W_in = input.size(3);
  int64_t H_out = output.size(2);
  int64_t W_out = output.size(3);

  InterpolationWeights horiz_w = compute_weights(W_in, W_out, C);
  InterpolationWeights vert_w = compute_weights(H_in, H_out, W_out * C);

  const int64_t row_bytes = W_out * C;
  const int64_t ring_capacity = vert_w.kmax;

  auto ring_buf_tensor = at::empty({ring_capacity * row_bytes}, input.options());
  uint8_t* ring_buf = ring_buf_tensor.data_ptr<uint8_t>();

  const uint8_t* input_ptr = input.const_data_ptr<uint8_t>();
  uint8_t* output_ptr = output.data_ptr<uint8_t>();
  const int64_t in_row_stride = W_in * C;

  // Precompute row pointer storage (reused each output row)
  std::vector<const uint8_t*> row_ptrs(ring_capacity);

  int64_t last_computed_row = -1;

  for (int64_t yy = 0; yy < H_out; yy++) {
    int64_t first_needed = vert_w.xmin[yy] / row_bytes;
    int64_t num_needed = vert_w.xsize[yy];
    int64_t last_needed = first_needed + num_needed - 1;

    // Horizontally scale only the new input rows into the ring buffer
    for (int64_t row = last_computed_row + 1; row <= last_needed; row++) {
      NeonResampleHorizontalConvolution8u(
          ring_buf + (row % ring_capacity) * row_bytes,
          W_out,
          input_ptr + row * in_row_stride,
          W_in,
          horiz_w.xmin.data(), horiz_w.xsize.data(),
          horiz_w.weights.data(), horiz_w.kmax,
          horiz_w.precision, C);
    }
    last_computed_row = std::max(last_computed_row, last_needed);

    // Build row pointers for vertical conv
    for (int64_t i = 0; i < num_needed; i++) {
      row_ptrs[i] = ring_buf + ((first_needed + i) % ring_capacity) * row_bytes;
    }

    // Vertical convolution reading from ring buffer
    NeonRingVerticalConvolution8u(
        output_ptr + yy * row_bytes,
        row_ptrs.data(),
        W_out,
        num_needed,
        &vert_w.weights[yy * vert_w.kmax],
        vert_w.precision,
        C);
  }
}

#endif // __aarch64__

} // anonymous namespace

at::Tensor upsample_bilinear2d_aa_ring_neon(
    const at::Tensor& input,
    at::IntArrayRef output_size) {

  TORCH_CHECK(input.dim() == 4, "Expected 4D input");
  TORCH_CHECK(output_size.size() == 2, "Expected 2D output size");
  TORCH_CHECK(input.device().is_cpu(), "Expected CPU tensor");
  TORCH_CHECK(input.scalar_type() == at::kByte, "Expected uint8 input");
  TORCH_CHECK(input.size(0) == 1, "Only batch size 1 supported");
  TORCH_CHECK(input.size(1) == 3, "Only 3 channels (RGB) supported");

  auto out_h = output_size[0];
  auto out_w = output_size[1];

  at::Tensor input_cl = input.contiguous(at::MemoryFormat::ChannelsLast);

  auto output = at::empty(
      {1, 3, out_h, out_w},
      input_cl.options().memory_format(at::MemoryFormat::ChannelsLast));

#if defined(__aarch64__)
  ring_neon_bilinear_uint8(input_cl, output);
#else
  TORCH_CHECK(false, "NEON not available on this platform");
#endif

  if (!input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return output.contiguous();
  }
  return output;
}

TORCH_LIBRARY(ring_neon_interpolate, m) {
  m.def("upsample_bilinear2d_aa(Tensor input, int[] output_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(ring_neon_interpolate, CPU, m) {
  m.impl("upsample_bilinear2d_aa", &upsample_bilinear2d_aa_ring_neon);
}

PYBIND11_MODULE(_C_ring_neon, m) {
  m.attr("ops") = py::module::import("torch").attr("ops");
}
