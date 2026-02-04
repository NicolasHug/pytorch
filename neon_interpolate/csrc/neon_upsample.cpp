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

constexpr int kPrecision = 12;  // Fixed-point precision for int16 weights

// Bilinear filter for antialiasing (triangle filter)
inline double aa_filter(double x) {
  x = std::abs(x);
  if (x < 1.0) {
    return 1.0 - x;
  }
  return 0.0;
}

// Compute scale factor: input_size / output_size
inline double compute_scale(int64_t in_size, int64_t out_size) {
  return static_cast<double>(in_size) / static_cast<double>(out_size);
}

struct InterpolationWeights {
  std::vector<int64_t> xmin;   // Input start index (byte offset) for each output pixel
  std::vector<int64_t> xsize;  // Number of input pixels to use for each output pixel
  std::vector<int16_t> weights; // Flattened weights for all output pixels
  int kmax;                     // Maximum interpolation size
  unsigned int precision;       // Weight precision
};

// Match the reference implementation in UpSampleKernel.cpp exactly
InterpolationWeights compute_weights(
    int64_t in_size,
    int64_t out_size,
    int64_t stride) {

  InterpolationWeights result;
  double scale = compute_scale(in_size, out_size);

  constexpr int interp_size = 2;  // Bilinear uses 2 input pixels
  double support;
  int max_interp_size;

  // Antialias is always true here
  support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
  max_interp_size = static_cast<int>(std::ceil(support) * 2 + 1);

  // Align to int32 boundary for efficient SIMD loads
  while (max_interp_size % 2 != 0) {
    max_interp_size += 1;
  }

  result.xmin.resize(out_size);
  result.xsize.resize(out_size);
  result.weights.resize(out_size * max_interp_size, 0);
  result.kmax = max_interp_size;

  double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  double wt_max = 0.0;

  for (int64_t i = 0; i < out_size; i++) {
    // Match reference: center = scale * (i + 0.5)
    double center = scale * (i + 0.5);

    int64_t xmin_val = std::max(
        static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
    int64_t xsize_val = std::min(
        static_cast<int64_t>(center + support + 0.5), in_size) - xmin_val;
    xsize_val = std::max(xsize_val, static_cast<int64_t>(0));
    xsize_val = std::min(xsize_val, static_cast<int64_t>(max_interp_size));

    // Compute weights using the same formula as reference
    std::vector<double> wt_f(max_interp_size, 0.0);
    double wt_sum = 0.0;
    for (int64_t j = 0; j < xsize_val; j++) {
      // Match reference: filter_fn((j + xmin - center + 0.5) * invscale)
      double w = aa_filter((j + xmin_val - center + 0.5) * invscale);
      wt_f[j] = w;
      wt_sum += w;
    }

    // Normalize weights
    if (wt_sum != 0.0) {
      for (int64_t j = 0; j < xsize_val; j++) {
        wt_f[j] /= wt_sum;
        wt_max = std::max(wt_max, wt_f[j]);
      }
    }

    result.xmin[i] = xmin_val * stride;
    result.xsize[i] = xsize_val;

    // Convert to int16 with precision
    for (int64_t j = 0; j < xsize_val; j++) {
      double v = wt_f[j] * (1 << kPrecision);
      result.weights[i * max_interp_size + j] = static_cast<int16_t>(v < 0 ? v - 0.5 : v + 0.5);
    }
    // Zero out remaining
    for (int64_t j = xsize_val; j < max_interp_size; j++) {
      result.weights[i * max_interp_size + j] = 0;
    }
  }

  result.precision = kPrecision;
  return result;
}

#if defined(__aarch64__)

// NEON implementation of madd_epi16 equivalent
// Multiply pairs of int16 and add adjacent results to get int32
inline int32x4_t neon_madd_s16(int16x8_t a, int16x8_t b) {
  // Multiply low and high halves
  int32x4_t prod_low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
  int32x4_t prod_high = vmull_s16(vget_high_s16(a), vget_high_s16(b));

  // Add adjacent pairs: [a0*b0 + a1*b1, a2*b2 + a3*b3]
  return vpaddq_s32(prod_low, prod_high);
}

// Vertical convolution for a single output row
void NeonResampleVerticalConvolution8u(
    uint8_t* lineOut,
    const uint8_t* lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {

  const int64_t stride = num_channels;
  const int64_t data_size = xsize * stride;

  const int32_t initial_val = 1 << (coefs_precision - 1);
  const int32x4_t initial = vdupq_n_s32(initial_val);

  int64_t j = 0;

  // Process 16 bytes at a time
  for (; j + 16 <= data_size; j += 16) {
    int32x4_t sss0 = initial;
    int32x4_t sss1 = initial;
    int32x4_t sss2 = initial;
    int32x4_t sss3 = initial;

    const uint8_t* lineIn_min = lineIn + j + ids_min;

    // Process 2 weights at a time
    int64_t i = 0;
    for (; i + 1 < ids_size; i += 2) {
      int16_t w0 = k[i];
      int16_t w1 = k[i + 1];
      int16x8_t mmk = {w0, w1, w0, w1, w0, w1, w0, w1};

      // Load 16 bytes from two rows
      uint8x16_t src1 = vld1q_u8(lineIn_min + i * data_size);
      uint8x16_t src2 = vld1q_u8(lineIn_min + (i + 1) * data_size);

      // Interleave bytes from src1 and src2
      uint8x16x2_t interleaved = vzipq_u8(src1, src2);

      // Process first 8 interleaved bytes
      uint8x8_t inter_lo_lo = vget_low_u8(interleaved.val[0]);
      uint8x8_t inter_lo_hi = vget_high_u8(interleaved.val[0]);
      uint8x8_t inter_hi_lo = vget_low_u8(interleaved.val[1]);
      uint8x8_t inter_hi_hi = vget_high_u8(interleaved.val[1]);

      // Widen to 16-bit
      int16x8_t pix0 = vreinterpretq_s16_u16(vmovl_u8(inter_lo_lo));
      int16x8_t pix1 = vreinterpretq_s16_u16(vmovl_u8(inter_lo_hi));
      int16x8_t pix2 = vreinterpretq_s16_u16(vmovl_u8(inter_hi_lo));
      int16x8_t pix3 = vreinterpretq_s16_u16(vmovl_u8(inter_hi_hi));

      // Multiply-accumulate
      sss0 = vaddq_s32(sss0, neon_madd_s16(pix0, mmk));
      sss1 = vaddq_s32(sss1, neon_madd_s16(pix1, mmk));
      sss2 = vaddq_s32(sss2, neon_madd_s16(pix2, mmk));
      sss3 = vaddq_s32(sss3, neon_madd_s16(pix3, mmk));
    }

    // Handle remaining single weight
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      int16x8_t mmk = vdupq_n_s16(w);

      uint8x16_t src = vld1q_u8(lineIn_min + i * data_size);

      // Widen to 16-bit
      uint8x8_t src_lo = vget_low_u8(src);
      uint8x8_t src_hi = vget_high_u8(src);

      int16x8_t pix_lo = vreinterpretq_s16_u16(vmovl_u8(src_lo));
      int16x8_t pix_hi = vreinterpretq_s16_u16(vmovl_u8(src_hi));

      // For single weight: multiply each element and widen to 32-bit
      int32x4_t prod0 = vmull_s16(vget_low_s16(pix_lo), vget_low_s16(mmk));
      int32x4_t prod1 = vmull_s16(vget_high_s16(pix_lo), vget_high_s16(mmk));
      int32x4_t prod2 = vmull_s16(vget_low_s16(pix_hi), vget_low_s16(mmk));
      int32x4_t prod3 = vmull_s16(vget_high_s16(pix_hi), vget_high_s16(mmk));

      sss0 = vaddq_s32(sss0, prod0);
      sss1 = vaddq_s32(sss1, prod1);
      sss2 = vaddq_s32(sss2, prod2);
      sss3 = vaddq_s32(sss3, prod3);
    }

    // Shift right by precision
    int32x4_t shift = vdupq_n_s32(-static_cast<int32_t>(coefs_precision));
    sss0 = vshlq_s32(sss0, shift);
    sss1 = vshlq_s32(sss1, shift);
    sss2 = vshlq_s32(sss2, shift);
    sss3 = vshlq_s32(sss3, shift);

    // Pack to 16-bit with saturation
    int16x4_t narrow0 = vqmovn_s32(sss0);
    int16x4_t narrow1 = vqmovn_s32(sss1);
    int16x4_t narrow2 = vqmovn_s32(sss2);
    int16x4_t narrow3 = vqmovn_s32(sss3);

    int16x8_t narrow_lo = vcombine_s16(narrow0, narrow1);
    int16x8_t narrow_hi = vcombine_s16(narrow2, narrow3);

    // Pack to 8-bit unsigned with saturation
    uint8x8_t result_lo = vqmovun_s16(narrow_lo);
    uint8x8_t result_hi = vqmovun_s16(narrow_hi);

    // Store
    vst1_u8(lineOut + j, result_lo);
    vst1_u8(lineOut + j + 8, result_hi);
  }

  // Scalar fallback for remaining pixels
  for (; j < data_size; j++) {
    int32_t sss = initial_val;
    const uint8_t* lineIn_min = lineIn + j + ids_min;

    for (int64_t i = 0; i < ids_size; i++) {
      sss += k[i] * static_cast<int32_t>(lineIn_min[i * data_size]);
    }

    sss >>= coefs_precision;
    sss = std::max(0, std::min(255, sss));
    lineOut[j] = static_cast<uint8_t>(sss);
  }
}

// Horizontal convolution for a single output row
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

    int32x4_t sss = vdupq_n_s32(initial_val);

    const uint8_t* lineIn_min = lineIn + ids_min;

    // Process 2 pixels at a time
    int64_t i = 0;
    for (; i + 1 < ids_size; i += 2) {
      int16_t w0 = k[i];
      int16_t w1 = k[i + 1];

      // For RGB (stride=3), load 6 bytes (2 pixels), expand to allow processing
      // For simplicity, we'll use scalar for the horizontal pass when stride=3

      // Load two pixels
      const uint8_t* p0 = lineIn_min + stride * i;
      const uint8_t* p1 = lineIn_min + stride * (i + 1);

      // Manual interleave for RGB
      if (num_channels == 3) {
        // Scalar path for RGB to handle non-aligned access
        for (int c = 0; c < 3; c++) {
          int32_t val = w0 * p0[c] + w1 * p1[c];
          sss = vsetq_lane_s32(vgetq_lane_s32(sss, c) + val, sss, c);
        }
      } else {
        // For 4 channels we can use NEON more efficiently
        uint8x8_t src = {p0[0], p1[0], p0[1], p1[1], p0[2], p1[2], p0[3], p1[3]};
        int16x8_t pix = vreinterpretq_s16_u16(vmovl_u8(src));
        int16x8_t mmk = {w0, w1, w0, w1, w0, w1, w0, w1};
        sss = vaddq_s32(sss, neon_madd_s16(pix, mmk));
      }
    }

    // Handle remaining single pixel
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p = lineIn_min + stride * i;

      for (int c = 0; c < num_channels; c++) {
        int32_t val = w * p[c];
        sss = vsetq_lane_s32(vgetq_lane_s32(sss, c) + val, sss, c);
      }
    }

    // Shift right by precision
    int32x4_t shift = vdupq_n_s32(-static_cast<int32_t>(coefs_precision));
    sss = vshlq_s32(sss, shift);

    // Clamp and store
    uint8_t* out = lineOut + stride * out_x;
    for (int c = 0; c < num_channels; c++) {
      int32_t v = vgetq_lane_s32(sss, c);
      v = std::max(0, std::min(255, v));
      out[c] = static_cast<uint8_t>(v);
    }
  }
}

void upsample_neon_bilinear_uint8(
    const at::Tensor& input,
    const at::Tensor& output) {

  auto batch_size = input.size(0);
  auto num_channels = input.size(1);
  auto yin = input.size(2);
  auto xin = input.size(3);
  auto yout = output.size(2);
  auto xout = output.size(3);

  TORCH_CHECK(num_channels == 3, "Only 3 channels (RGB) supported");
  TORCH_CHECK(input.is_contiguous(at::MemoryFormat::ChannelsLast),
              "Input must be channels_last");
  TORCH_CHECK(output.is_contiguous(at::MemoryFormat::ChannelsLast),
              "Output must be channels_last");

  // Compute weights for horizontal and vertical passes
  InterpolationWeights horiz_weights = compute_weights(xin, xout, num_channels);
  InterpolationWeights vert_weights = compute_weights(yin, yout, xout * num_channels);

  // Intermediate buffer for horizontal pass output
  auto buffer = at::empty({yin, xout, num_channels}, input.options());

  const uint8_t* input_ptr = input.const_data_ptr<uint8_t>();
  uint8_t* output_ptr = output.data_ptr<uint8_t>();
  uint8_t* buffer_ptr = buffer.data_ptr<uint8_t>();

  const int64_t in_stride_batch = yin * xin * num_channels;
  const int64_t out_stride_batch = yout * xout * num_channels;
  const int64_t in_row_stride = xin * num_channels;
  const int64_t buffer_row_stride = xout * num_channels;
  const int64_t out_row_stride = xout * num_channels;

  for (int64_t b = 0; b < batch_size; b++) {
    const uint8_t* input_batch = input_ptr + b * in_stride_batch;
    uint8_t* output_batch = output_ptr + b * out_stride_batch;

    // Horizontal pass: input -> buffer
    for (int64_t y = 0; y < yin; y++) {
      const uint8_t* in_row = input_batch + y * in_row_stride;
      uint8_t* buf_row = buffer_ptr + y * buffer_row_stride;

      NeonResampleHorizontalConvolution8u(
          buf_row,
          xout,
          in_row,
          xin,
          horiz_weights.xmin.data(),
          horiz_weights.xsize.data(),
          horiz_weights.weights.data(),
          horiz_weights.kmax,
          horiz_weights.precision,
          num_channels);
    }

    // Vertical pass: buffer -> output
    for (int64_t y = 0; y < yout; y++) {
      uint8_t* out_row = output_batch + y * out_row_stride;
      const int16_t* k = &vert_weights.weights[y * vert_weights.kmax];
      int64_t ids_min = vert_weights.xmin[y];
      int64_t ids_size = vert_weights.xsize[y];

      NeonResampleVerticalConvolution8u(
          out_row,
          buffer_ptr,
          xout,
          ids_min,
          ids_size,
          k,
          vert_weights.precision,
          num_channels);
    }
  }
}

#endif // __aarch64__

} // anonymous namespace

at::Tensor upsample_bilinear2d_aa_neon(
    const at::Tensor& input,
    at::IntArrayRef output_size) {

  TORCH_CHECK(input.dim() == 4, "Expected 4D input");
  TORCH_CHECK(output_size.size() == 2, "Expected 2D output size");
  TORCH_CHECK(input.device().is_cpu(), "Expected CPU tensor");
  TORCH_CHECK(input.scalar_type() == at::kByte, "Expected uint8 input");

  auto batch = input.size(0);
  auto channels = input.size(1);
  auto out_h = output_size[0];
  auto out_w = output_size[1];

  TORCH_CHECK(channels == 3, "Only 3 channels (RGB) supported");

  // Convert to channels_last if needed
  at::Tensor input_cl = input.contiguous(at::MemoryFormat::ChannelsLast);

  auto output = at::empty(
      {batch, channels, out_h, out_w},
      input_cl.options().memory_format(at::MemoryFormat::ChannelsLast));

#if defined(__aarch64__)
  upsample_neon_bilinear_uint8(input_cl, output);
#else
  TORCH_CHECK(false, "NEON not available on this platform");
#endif

  // Convert back to original memory format if needed
  if (!input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return output.contiguous();
  }
  return output;
}

TORCH_LIBRARY(neon_interpolate, m) {
  m.def("upsample_bilinear2d_aa(Tensor input, int[] output_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(neon_interpolate, CPU, m) {
  m.impl("upsample_bilinear2d_aa", &upsample_bilinear2d_aa_neon);
}

PYBIND11_MODULE(_C, m) {
  m.attr("ops") = py::module::import("torch").attr("ops");
}
