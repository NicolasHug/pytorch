/*
SVE2 implementation of bilinear interpolation with antialiasing.
This is designed for ARM Neoverse-V2 and similar processors with SVE2 support.

The implementation follows the same structure as the AVX2 implementation in
aten/src/ATen/native/cpu/UpSampleKernelAVXAntialias.h
*/

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif

namespace {

constexpr int kPrecision = 12; // Fixed-point precision for int16 weights

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
  std::vector<int64_t>
      xmin; // Input start index (byte offset) for each output pixel
  std::vector<int64_t>
      xsize; // Number of input pixels to use for each output pixel
  std::vector<int16_t> weights; // Flattened weights for all output pixels
  int kmax; // Maximum interpolation size
  unsigned int precision; // Weight precision
};

// Compute interpolation weights - matches PyTorch's reference implementation
InterpolationWeights compute_weights(
    int64_t in_size,
    int64_t out_size,
    int64_t stride) {
  InterpolationWeights result;
  double scale = compute_scale(in_size, out_size);

  constexpr int interp_size = 2; // Bilinear uses 2 input pixels
  double support;
  int max_interp_size;

  // Antialias is always true here
  support = (scale >= 1.0) ? (interp_size * 0.5) * scale : interp_size * 0.5;
  max_interp_size = static_cast<int>(std::ceil(support) * 2 + 1);

  // Align to 2 for efficient SIMD pair processing (matching AVX2's madd
  // pattern)
  while (max_interp_size % 2 != 0) {
    max_interp_size += 1;
  }

  result.xmin.resize(out_size);
  result.xsize.resize(out_size);
  result.weights.resize(out_size * max_interp_size, 0);
  result.kmax = max_interp_size;

  double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;

  for (int64_t i = 0; i < out_size; i++) {
    // Match reference: center = scale * (i + 0.5)
    double center = scale * (i + 0.5);

    int64_t xmin_val = std::max(
        static_cast<int64_t>(center - support + 0.5), static_cast<int64_t>(0));
    int64_t xsize_val =
        std::min(static_cast<int64_t>(center + support + 0.5), in_size) -
        xmin_val;
    xsize_val = std::max(xsize_val, static_cast<int64_t>(0));
    xsize_val = std::min(xsize_val, static_cast<int64_t>(max_interp_size));

    // Compute weights using the same formula as reference
    std::vector<double> wt_f(max_interp_size, 0.0);
    double wt_sum = 0.0;
    for (int64_t j = 0; j < xsize_val; j++) {
      double w = aa_filter((j + xmin_val - center + 0.5) * invscale);
      wt_f[j] = w;
      wt_sum += w;
    }

    // Normalize weights
    if (wt_sum != 0.0) {
      for (int64_t j = 0; j < xsize_val; j++) {
        wt_f[j] /= wt_sum;
      }
    }

    result.xmin[i] = xmin_val * stride;
    result.xsize[i] = xsize_val;

    // Convert to int16 with precision
    for (int64_t j = 0; j < xsize_val; j++) {
      double v = wt_f[j] * (1 << kPrecision);
      result.weights[i * max_interp_size + j] =
          static_cast<int16_t>(v < 0 ? v - 0.5 : v + 0.5);
    }
    // Zero out remaining
    for (int64_t j = xsize_val; j < max_interp_size; j++) {
      result.weights[i * max_interp_size + j] = 0;
    }
  }

  result.precision = kPrecision;
  return result;
}

#if defined(__ARM_FEATURE_SVE2)

// ============================================================================
// SVE2 VERTICAL CONVOLUTION
// Process as many bytes as the vector length allows per iteration
// Each weight applies to all pixels in a row, so we can vectorize across X
// ============================================================================
void SVE2ResampleVerticalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    const uint8_t* C10_RESTRICT lineIn,
    int64_t xsize,
    int64_t ids_min,
    int64_t ids_size,
    const int16_t* k,
    unsigned int coefs_precision,
    int64_t num_channels) {
  const int64_t data_size = xsize * num_channels;
  const int32_t initial_val = 1 << (coefs_precision - 1);

  // Use NEON for vertical convolution (same 128-bit width as SVE on
  // Neoverse-V2) This is simpler and avoids SVE predication overhead
  int64_t j = 0;

  // Process 16 bytes at a time
  for (; j + 16 <= data_size; j += 16) {
    // Initialize 32-bit accumulators (16 bytes -> 16 int32s)
    int32x4_t acc0 = vdupq_n_s32(initial_val);
    int32x4_t acc1 = vdupq_n_s32(initial_val);
    int32x4_t acc2 = vdupq_n_s32(initial_val);
    int32x4_t acc3 = vdupq_n_s32(initial_val);

    const uint8_t* lineIn_pos = lineIn + j + ids_min;

    for (int64_t i = 0; i < ids_size; i++) {
      int16_t w = k[i];
      int16x4_t weight = vdup_n_s16(w);

      // Load 16 bytes from this row
      uint8x16_t src = vld1q_u8(lineIn_pos + i * data_size);

      // Widen to 16-bit
      uint8x8_t src_lo = vget_low_u8(src);
      uint8x8_t src_hi = vget_high_u8(src);
      int16x8_t pix_lo = vreinterpretq_s16_u16(vmovl_u8(src_lo));
      int16x8_t pix_hi = vreinterpretq_s16_u16(vmovl_u8(src_hi));

      // Multiply and accumulate to 32-bit
      acc0 = vmlal_s16(acc0, vget_low_s16(pix_lo), weight);
      acc1 = vmlal_s16(acc1, vget_high_s16(pix_lo), weight);
      acc2 = vmlal_s16(acc2, vget_low_s16(pix_hi), weight);
      acc3 = vmlal_s16(acc3, vget_high_s16(pix_hi), weight);
    }

    // Shift right
    int32x4_t shift = vdupq_n_s32(-static_cast<int32_t>(coefs_precision));
    acc0 = vshlq_s32(acc0, shift);
    acc1 = vshlq_s32(acc1, shift);
    acc2 = vshlq_s32(acc2, shift);
    acc3 = vshlq_s32(acc3, shift);

    // Narrow with saturation: 32->16->8
    int16x4_t narrow0 = vqmovn_s32(acc0);
    int16x4_t narrow1 = vqmovn_s32(acc1);
    int16x4_t narrow2 = vqmovn_s32(acc2);
    int16x4_t narrow3 = vqmovn_s32(acc3);
    int16x8_t narrow_lo = vcombine_s16(narrow0, narrow1);
    int16x8_t narrow_hi = vcombine_s16(narrow2, narrow3);
    uint8x8_t result_lo = vqmovun_s16(narrow_lo);
    uint8x8_t result_hi = vqmovun_s16(narrow_hi);

    // Store
    vst1_u8(lineOut + j, result_lo);
    vst1_u8(lineOut + j + 8, result_hi);
  }

  // Handle remaining bytes with scalar code
  for (; j < data_size; j++) {
    int32_t acc = initial_val;
    const uint8_t* lineIn_pos = lineIn + j + ids_min;

    for (int64_t i = 0; i < ids_size; i++) {
      acc += static_cast<int32_t>(lineIn_pos[i * data_size]) * k[i];
    }

    acc >>= coefs_precision;
    lineOut[j] = static_cast<uint8_t>(std::max(0, std::min(255, acc)));
  }
}

// ============================================================================
// SVE2 HORIZONTAL CONVOLUTION - Processing 4 rows simultaneously
// Uses NEON intrinsics for the inner loop (SVE2 has same width on Neoverse-V2)
// Key optimization: Process 4 weights at a time with widening
// multiply-accumulate
// ============================================================================
void SVE2ResampleHorizontalConvolution8u4x(
    uint8_t* C10_RESTRICT lineOut0,
    uint8_t* C10_RESTRICT lineOut1,
    uint8_t* C10_RESTRICT lineOut2,
    uint8_t* C10_RESTRICT lineOut3,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn0,
    const uint8_t* C10_RESTRICT lineIn1,
    const uint8_t* C10_RESTRICT lineIn2,
    const uint8_t* C10_RESTRICT lineIn3,
    int64_t in_xsize,
    const int64_t* idx_ptr_xmin,
    const int64_t* idx_ptr_size,
    const int16_t* kk,
    int kmax,
    unsigned int coefs_precision,
    int64_t num_channels) {
  const int64_t stride = num_channels;
  const int32_t initial_val = 1 << (coefs_precision - 1);

  // Shuffle mask to convert RGBRGBRGBRGB... to RR00GG00BB00 pattern for
  // madd-style We load 12 bytes (4 RGB pixels) and want to extract R, G, B
  // separately Use table lookup: indices for extracting R values: 0, 3, 6, 9
  // from 12-byte input

  for (int64_t out_x = 0; out_x < out_xsize; out_x++) {
    const int64_t ids_min = idx_ptr_xmin[out_x];
    const int64_t ids_size = idx_ptr_size[out_x];
    const int16_t* k = &kk[out_x * kmax];

    const uint8_t* lineIn0_min = lineIn0 + ids_min;
    const uint8_t* lineIn1_min = lineIn1 + ids_min;
    const uint8_t* lineIn2_min = lineIn2 + ids_min;
    const uint8_t* lineIn3_min = lineIn3 + ids_min;

    // Scalar accumulators - horizontal convolution fundamentally needs
    // reduction Keep scalar for correctness; SIMD doesn't help much for small
    // ids_size
    int32_t acc0_r = initial_val, acc0_g = initial_val, acc0_b = initial_val;
    int32_t acc1_r = initial_val, acc1_g = initial_val, acc1_b = initial_val;
    int32_t acc2_r = initial_val, acc2_g = initial_val, acc2_b = initial_val;
    int32_t acc3_r = initial_val, acc3_g = initial_val, acc3_b = initial_val;

    int64_t i = 0;

    // Process 4 weights at a time using NEON - good for larger kernels
    // This avoids per-pixel horizontal reduction overhead when ids_size >= 4
    for (; i + 4 <= ids_size; i += 4) {
      int16x4_t w4 = vld1_s16(&k[i]);

      // Row 0: Load 4 RGB pixels (12 bytes) - vld3 deinterleaves automatically
      uint8x8x3_t rgb0 = vld3_u8(lineIn0_min + stride * i);
      int16x4_t r0 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb0.val[0])));
      int16x4_t g0 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb0.val[1])));
      int16x4_t b0 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb0.val[2])));
      int32x4_t prod0_r = vmull_s16(r0, w4);
      int32x4_t prod0_g = vmull_s16(g0, w4);
      int32x4_t prod0_b = vmull_s16(b0, w4);

      // Row 1
      uint8x8x3_t rgb1 = vld3_u8(lineIn1_min + stride * i);
      int16x4_t r1 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb1.val[0])));
      int16x4_t g1 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb1.val[1])));
      int16x4_t b1 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb1.val[2])));
      int32x4_t prod1_r = vmull_s16(r1, w4);
      int32x4_t prod1_g = vmull_s16(g1, w4);
      int32x4_t prod1_b = vmull_s16(b1, w4);

      // Row 2
      uint8x8x3_t rgb2 = vld3_u8(lineIn2_min + stride * i);
      int16x4_t r2 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb2.val[0])));
      int16x4_t g2 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb2.val[1])));
      int16x4_t b2 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb2.val[2])));
      int32x4_t prod2_r = vmull_s16(r2, w4);
      int32x4_t prod2_g = vmull_s16(g2, w4);
      int32x4_t prod2_b = vmull_s16(b2, w4);

      // Row 3
      uint8x8x3_t rgb3 = vld3_u8(lineIn3_min + stride * i);
      int16x4_t r3 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb3.val[0])));
      int16x4_t g3 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb3.val[1])));
      int16x4_t b3 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(rgb3.val[2])));
      int32x4_t prod3_r = vmull_s16(r3, w4);
      int32x4_t prod3_g = vmull_s16(g3, w4);
      int32x4_t prod3_b = vmull_s16(b3, w4);

      // Horizontal sum and accumulate (unavoidable for horizontal convolution)
      acc0_r += vaddvq_s32(prod0_r);
      acc0_g += vaddvq_s32(prod0_g);
      acc0_b += vaddvq_s32(prod0_b);
      acc1_r += vaddvq_s32(prod1_r);
      acc1_g += vaddvq_s32(prod1_g);
      acc1_b += vaddvq_s32(prod1_b);
      acc2_r += vaddvq_s32(prod2_r);
      acc2_g += vaddvq_s32(prod2_g);
      acc2_b += vaddvq_s32(prod2_b);
      acc3_r += vaddvq_s32(prod3_r);
      acc3_g += vaddvq_s32(prod3_g);
      acc3_b += vaddvq_s32(prod3_b);
    }

    // Process remaining 1-3 weights with scalar code
    for (; i < ids_size; i++) {
      int16_t w = k[i];

      const uint8_t* p0 = lineIn0_min + stride * i;
      const uint8_t* p1 = lineIn1_min + stride * i;
      const uint8_t* p2 = lineIn2_min + stride * i;
      const uint8_t* p3 = lineIn3_min + stride * i;

      acc0_r += p0[0] * w;
      acc0_g += p0[1] * w;
      acc0_b += p0[2] * w;
      acc1_r += p1[0] * w;
      acc1_g += p1[1] * w;
      acc1_b += p1[2] * w;
      acc2_r += p2[0] * w;
      acc2_g += p2[1] * w;
      acc2_b += p2[2] * w;
      acc3_r += p3[0] * w;
      acc3_g += p3[1] * w;
      acc3_b += p3[2] * w;
    }

    // Shift, clamp, and store
    uint8_t* o0 = lineOut0 + stride * out_x;
    uint8_t* o1 = lineOut1 + stride * out_x;
    uint8_t* o2 = lineOut2 + stride * out_x;
    uint8_t* o3 = lineOut3 + stride * out_x;

    o0[0] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc0_r >> coefs_precision)));
    o0[1] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc0_g >> coefs_precision)));
    o0[2] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc0_b >> coefs_precision)));
    o1[0] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc1_r >> coefs_precision)));
    o1[1] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc1_g >> coefs_precision)));
    o1[2] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc1_b >> coefs_precision)));
    o2[0] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc2_r >> coefs_precision)));
    o2[1] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc2_g >> coefs_precision)));
    o2[2] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc2_b >> coefs_precision)));
    o3[0] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc3_r >> coefs_precision)));
    o3[1] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc3_g >> coefs_precision)));
    o3[2] = static_cast<uint8_t>(
        std::max(0, std::min(255, acc3_b >> coefs_precision)));
  }
}

// ============================================================================
// SVE2 HORIZONTAL CONVOLUTION - Processing 1 row
// ============================================================================
void SVE2ResampleHorizontalConvolution8u(
    uint8_t* C10_RESTRICT lineOut,
    int64_t out_xsize,
    const uint8_t* C10_RESTRICT lineIn,
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

    int32_t acc[4] = {initial_val, initial_val, initial_val, 0};

    int64_t i = 0;

    // Process 2 weights at a time
    for (; i + 2 <= ids_size; i += 2) {
      int16_t w0 = k[i];
      int16_t w1 = k[i + 1];

      const uint8_t* p0 = lineIn_min + stride * i;
      const uint8_t* p1 = lineIn_min + stride * (i + 1);

      acc[0] += p0[0] * w0 + p1[0] * w1;
      acc[1] += p0[1] * w0 + p1[1] * w1;
      acc[2] += p0[2] * w0 + p1[2] * w1;
    }

    // Handle remaining single weight
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p = lineIn_min + stride * i;

      acc[0] += p[0] * w;
      acc[1] += p[1] * w;
      acc[2] += p[2] * w;
    }

    // Shift, clamp, and store
    uint8_t* out = lineOut + stride * out_x;
    for (int c = 0; c < 3; c++) {
      int32_t v = acc[c] >> coefs_precision;
      out[c] = static_cast<uint8_t>(std::max(0, std::min(255, v)));
    }
  }
}

void upsample_sve2_bilinear_uint8(
    const at::Tensor& input,
    const at::Tensor& output) {
  auto batch_size = input.size(0);
  auto num_channels = input.size(1);
  auto yin = input.size(2);
  auto xin = input.size(3);
  auto yout = output.size(2);
  auto xout = output.size(3);

  TORCH_CHECK(num_channels == 3, "Only 3 channels (RGB) supported");
  TORCH_CHECK(
      input.is_contiguous(at::MemoryFormat::ChannelsLast),
      "Input must be channels_last");
  TORCH_CHECK(
      output.is_contiguous(at::MemoryFormat::ChannelsLast),
      "Output must be channels_last");

  // Compute weights for horizontal and vertical passes
  InterpolationWeights horiz_weights = compute_weights(xin, xout, num_channels);
  InterpolationWeights vert_weights =
      compute_weights(yin, yout, xout * num_channels);

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

    // Horizontal pass: input -> buffer (process 4 rows at a time)
    int64_t y = 0;
    for (; y + 4 <= yin; y += 4) {
      const uint8_t* in_row0 = input_batch + y * in_row_stride;
      const uint8_t* in_row1 = input_batch + (y + 1) * in_row_stride;
      const uint8_t* in_row2 = input_batch + (y + 2) * in_row_stride;
      const uint8_t* in_row3 = input_batch + (y + 3) * in_row_stride;
      uint8_t* buf_row0 = buffer_ptr + y * buffer_row_stride;
      uint8_t* buf_row1 = buffer_ptr + (y + 1) * buffer_row_stride;
      uint8_t* buf_row2 = buffer_ptr + (y + 2) * buffer_row_stride;
      uint8_t* buf_row3 = buffer_ptr + (y + 3) * buffer_row_stride;

      SVE2ResampleHorizontalConvolution8u4x(
          buf_row0,
          buf_row1,
          buf_row2,
          buf_row3,
          xout,
          in_row0,
          in_row1,
          in_row2,
          in_row3,
          xin,
          horiz_weights.xmin.data(),
          horiz_weights.xsize.data(),
          horiz_weights.weights.data(),
          horiz_weights.kmax,
          horiz_weights.precision,
          num_channels);
    }
    // Handle remaining rows
    for (; y < yin; y++) {
      const uint8_t* in_row = input_batch + y * in_row_stride;
      uint8_t* buf_row = buffer_ptr + y * buffer_row_stride;

      SVE2ResampleHorizontalConvolution8u(
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

      SVE2ResampleVerticalConvolution8u(
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

#endif // __ARM_FEATURE_SVE2

} // anonymous namespace

at::Tensor upsample_bilinear2d_aa_sve2(
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

#if defined(__ARM_FEATURE_SVE2)
  upsample_sve2_bilinear_uint8(input_cl, output);
#else
  TORCH_CHECK(false, "SVE2 not available on this platform");
#endif

  // Convert back to original memory format if needed
  if (!input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return output.contiguous();
  }
  return output;
}

TORCH_LIBRARY(sve2_interpolate, m) {
  m.def("upsample_bilinear2d_aa(Tensor input, int[] output_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(sve2_interpolate, CPU, m) {
  m.impl("upsample_bilinear2d_aa", &upsample_bilinear2d_aa_sve2);
}

PYBIND11_MODULE(_C_sve2, m) {
  m.attr("ops") = py::module::import("torch").attr("ops");
}
