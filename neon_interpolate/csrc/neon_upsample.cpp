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

// Horizontal convolution processing 4 rows at once - maximizes weight reuse
void NeonResampleHorizontalConvolution8u4x(
    uint8_t* lineOut0,
    uint8_t* lineOut1,
    uint8_t* lineOut2,
    uint8_t* lineOut3,
    int64_t out_xsize,
    const uint8_t* lineIn0,
    const uint8_t* lineIn1,
    const uint8_t* lineIn2,
    const uint8_t* lineIn3,
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

    const uint8_t* lineIn0_min = lineIn0 + ids_min;
    const uint8_t* lineIn1_min = lineIn1 + ids_min;
    const uint8_t* lineIn2_min = lineIn2 + ids_min;
    const uint8_t* lineIn3_min = lineIn3 + ids_min;

    // Accumulators for 4 rows, R/G/B channels
    int32x4_t acc0_r = vdupq_n_s32(0), acc0_g = vdupq_n_s32(0), acc0_b = vdupq_n_s32(0);
    int32x4_t acc1_r = vdupq_n_s32(0), acc1_g = vdupq_n_s32(0), acc1_b = vdupq_n_s32(0);
    int32x4_t acc2_r = vdupq_n_s32(0), acc2_g = vdupq_n_s32(0), acc2_b = vdupq_n_s32(0);
    int32x4_t acc3_r = vdupq_n_s32(0), acc3_g = vdupq_n_s32(0), acc3_b = vdupq_n_s32(0);

    int64_t i = 0;

    // Process 8 pixels at a time - weights loaded once, applied to 4 rows
    for (; i + 8 <= ids_size; i += 8) {
      int16x8_t weights = vld1q_s16(&k[i]);
      int16x4_t w_lo = vget_low_s16(weights);
      int16x4_t w_hi = vget_high_s16(weights);

      // Row 0
      uint8x8x3_t rgb0 = vld3_u8(lineIn0_min + stride * i);
      int16x8_t r0 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[0]));
      int16x8_t g0 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[1]));
      int16x8_t b0 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[2]));
      acc0_r = vmlal_s16(acc0_r, vget_low_s16(r0), w_lo);
      acc0_r = vmlal_s16(acc0_r, vget_high_s16(r0), w_hi);
      acc0_g = vmlal_s16(acc0_g, vget_low_s16(g0), w_lo);
      acc0_g = vmlal_s16(acc0_g, vget_high_s16(g0), w_hi);
      acc0_b = vmlal_s16(acc0_b, vget_low_s16(b0), w_lo);
      acc0_b = vmlal_s16(acc0_b, vget_high_s16(b0), w_hi);

      // Row 1
      uint8x8x3_t rgb1 = vld3_u8(lineIn1_min + stride * i);
      int16x8_t r1 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[0]));
      int16x8_t g1 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[1]));
      int16x8_t b1 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[2]));
      acc1_r = vmlal_s16(acc1_r, vget_low_s16(r1), w_lo);
      acc1_r = vmlal_s16(acc1_r, vget_high_s16(r1), w_hi);
      acc1_g = vmlal_s16(acc1_g, vget_low_s16(g1), w_lo);
      acc1_g = vmlal_s16(acc1_g, vget_high_s16(g1), w_hi);
      acc1_b = vmlal_s16(acc1_b, vget_low_s16(b1), w_lo);
      acc1_b = vmlal_s16(acc1_b, vget_high_s16(b1), w_hi);

      // Row 2
      uint8x8x3_t rgb2 = vld3_u8(lineIn2_min + stride * i);
      int16x8_t r2 = vreinterpretq_s16_u16(vmovl_u8(rgb2.val[0]));
      int16x8_t g2 = vreinterpretq_s16_u16(vmovl_u8(rgb2.val[1]));
      int16x8_t b2 = vreinterpretq_s16_u16(vmovl_u8(rgb2.val[2]));
      acc2_r = vmlal_s16(acc2_r, vget_low_s16(r2), w_lo);
      acc2_r = vmlal_s16(acc2_r, vget_high_s16(r2), w_hi);
      acc2_g = vmlal_s16(acc2_g, vget_low_s16(g2), w_lo);
      acc2_g = vmlal_s16(acc2_g, vget_high_s16(g2), w_hi);
      acc2_b = vmlal_s16(acc2_b, vget_low_s16(b2), w_lo);
      acc2_b = vmlal_s16(acc2_b, vget_high_s16(b2), w_hi);

      // Row 3
      uint8x8x3_t rgb3 = vld3_u8(lineIn3_min + stride * i);
      int16x8_t r3 = vreinterpretq_s16_u16(vmovl_u8(rgb3.val[0]));
      int16x8_t g3 = vreinterpretq_s16_u16(vmovl_u8(rgb3.val[1]));
      int16x8_t b3 = vreinterpretq_s16_u16(vmovl_u8(rgb3.val[2]));
      acc3_r = vmlal_s16(acc3_r, vget_low_s16(r3), w_lo);
      acc3_r = vmlal_s16(acc3_r, vget_high_s16(r3), w_hi);
      acc3_g = vmlal_s16(acc3_g, vget_low_s16(g3), w_lo);
      acc3_g = vmlal_s16(acc3_g, vget_high_s16(g3), w_hi);
      acc3_b = vmlal_s16(acc3_b, vget_low_s16(b3), w_lo);
      acc3_b = vmlal_s16(acc3_b, vget_high_s16(b3), w_hi);
    }

    // Process 4 pixels at a time
    for (; i + 4 <= ids_size; i += 4) {
      int16x4_t w4 = vld1_s16(&k[i]);

      uint8x8x3_t rgb0 = vld3_u8(lineIn0_min + stride * i);
      acc0_r = vmlal_s16(acc0_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[0]))), w4);
      acc0_g = vmlal_s16(acc0_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[1]))), w4);
      acc0_b = vmlal_s16(acc0_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[2]))), w4);

      uint8x8x3_t rgb1 = vld3_u8(lineIn1_min + stride * i);
      acc1_r = vmlal_s16(acc1_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[0]))), w4);
      acc1_g = vmlal_s16(acc1_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[1]))), w4);
      acc1_b = vmlal_s16(acc1_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[2]))), w4);

      uint8x8x3_t rgb2 = vld3_u8(lineIn2_min + stride * i);
      acc2_r = vmlal_s16(acc2_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb2.val[0]))), w4);
      acc2_g = vmlal_s16(acc2_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb2.val[1]))), w4);
      acc2_b = vmlal_s16(acc2_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb2.val[2]))), w4);

      uint8x8x3_t rgb3 = vld3_u8(lineIn3_min + stride * i);
      acc3_r = vmlal_s16(acc3_r, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb3.val[0]))), w4);
      acc3_g = vmlal_s16(acc3_g, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb3.val[1]))), w4);
      acc3_b = vmlal_s16(acc3_b, vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb3.val[2]))), w4);
    }

    // Horizontal sums
    int32_t s0_r = vaddvq_s32(acc0_r) + initial_val;
    int32_t s0_g = vaddvq_s32(acc0_g) + initial_val;
    int32_t s0_b = vaddvq_s32(acc0_b) + initial_val;
    int32_t s1_r = vaddvq_s32(acc1_r) + initial_val;
    int32_t s1_g = vaddvq_s32(acc1_g) + initial_val;
    int32_t s1_b = vaddvq_s32(acc1_b) + initial_val;
    int32_t s2_r = vaddvq_s32(acc2_r) + initial_val;
    int32_t s2_g = vaddvq_s32(acc2_g) + initial_val;
    int32_t s2_b = vaddvq_s32(acc2_b) + initial_val;
    int32_t s3_r = vaddvq_s32(acc3_r) + initial_val;
    int32_t s3_g = vaddvq_s32(acc3_g) + initial_val;
    int32_t s3_b = vaddvq_s32(acc3_b) + initial_val;

    // Scalar cleanup
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p0 = lineIn0_min + stride * i;
      const uint8_t* p1 = lineIn1_min + stride * i;
      const uint8_t* p2 = lineIn2_min + stride * i;
      const uint8_t* p3 = lineIn3_min + stride * i;
      s0_r += w * p0[0]; s0_g += w * p0[1]; s0_b += w * p0[2];
      s1_r += w * p1[0]; s1_g += w * p1[1]; s1_b += w * p1[2];
      s2_r += w * p2[0]; s2_g += w * p2[1]; s2_b += w * p2[2];
      s3_r += w * p3[0]; s3_g += w * p3[1]; s3_b += w * p3[2];
    }

    // Store all 4 rows
    uint8_t* o0 = lineOut0 + stride * out_x;
    uint8_t* o1 = lineOut1 + stride * out_x;
    uint8_t* o2 = lineOut2 + stride * out_x;
    uint8_t* o3 = lineOut3 + stride * out_x;
    o0[0] = std::clamp(s0_r >> coefs_precision, 0, 255);
    o0[1] = std::clamp(s0_g >> coefs_precision, 0, 255);
    o0[2] = std::clamp(s0_b >> coefs_precision, 0, 255);
    o1[0] = std::clamp(s1_r >> coefs_precision, 0, 255);
    o1[1] = std::clamp(s1_g >> coefs_precision, 0, 255);
    o1[2] = std::clamp(s1_b >> coefs_precision, 0, 255);
    o2[0] = std::clamp(s2_r >> coefs_precision, 0, 255);
    o2[1] = std::clamp(s2_g >> coefs_precision, 0, 255);
    o2[2] = std::clamp(s2_b >> coefs_precision, 0, 255);
    o3[0] = std::clamp(s3_r >> coefs_precision, 0, 255);
    o3[1] = std::clamp(s3_g >> coefs_precision, 0, 255);
    o3[2] = std::clamp(s3_b >> coefs_precision, 0, 255);
  }
}

// Horizontal convolution processing 2 rows at once - amortizes weight loading
void NeonResampleHorizontalConvolution8u2x(
    uint8_t* lineOut0,
    uint8_t* lineOut1,
    int64_t out_xsize,
    const uint8_t* lineIn0,
    const uint8_t* lineIn1,
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

    const uint8_t* lineIn0_min = lineIn0 + ids_min;
    const uint8_t* lineIn1_min = lineIn1 + ids_min;

    // Accumulators for both rows, R/G/B channels
    int32x4_t acc0_r = vdupq_n_s32(0);
    int32x4_t acc0_g = vdupq_n_s32(0);
    int32x4_t acc0_b = vdupq_n_s32(0);
    int32x4_t acc1_r = vdupq_n_s32(0);
    int32x4_t acc1_g = vdupq_n_s32(0);
    int32x4_t acc1_b = vdupq_n_s32(0);

    int64_t i = 0;

    // Process 8 pixels at a time - weights loaded once, applied to both rows
    for (; i + 8 <= ids_size; i += 8) {
      // Load 8 weights ONCE
      int16x8_t weights = vld1q_s16(&k[i]);
      int16x4_t w_lo = vget_low_s16(weights);
      int16x4_t w_hi = vget_high_s16(weights);

      // Row 0: Load and deinterleave 8 RGB pixels
      uint8x8x3_t rgb0 = vld3_u8(lineIn0_min + stride * i);
      int16x8_t r0_16 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[0]));
      int16x8_t g0_16 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[1]));
      int16x8_t b0_16 = vreinterpretq_s16_u16(vmovl_u8(rgb0.val[2]));

      acc0_r = vmlal_s16(acc0_r, vget_low_s16(r0_16), w_lo);
      acc0_r = vmlal_s16(acc0_r, vget_high_s16(r0_16), w_hi);
      acc0_g = vmlal_s16(acc0_g, vget_low_s16(g0_16), w_lo);
      acc0_g = vmlal_s16(acc0_g, vget_high_s16(g0_16), w_hi);
      acc0_b = vmlal_s16(acc0_b, vget_low_s16(b0_16), w_lo);
      acc0_b = vmlal_s16(acc0_b, vget_high_s16(b0_16), w_hi);

      // Row 1: Load and deinterleave 8 RGB pixels (reuse weights)
      uint8x8x3_t rgb1 = vld3_u8(lineIn1_min + stride * i);
      int16x8_t r1_16 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[0]));
      int16x8_t g1_16 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[1]));
      int16x8_t b1_16 = vreinterpretq_s16_u16(vmovl_u8(rgb1.val[2]));

      acc1_r = vmlal_s16(acc1_r, vget_low_s16(r1_16), w_lo);
      acc1_r = vmlal_s16(acc1_r, vget_high_s16(r1_16), w_hi);
      acc1_g = vmlal_s16(acc1_g, vget_low_s16(g1_16), w_lo);
      acc1_g = vmlal_s16(acc1_g, vget_high_s16(g1_16), w_hi);
      acc1_b = vmlal_s16(acc1_b, vget_low_s16(b1_16), w_lo);
      acc1_b = vmlal_s16(acc1_b, vget_high_s16(b1_16), w_hi);
    }

    // Process 4 pixels at a time
    for (; i + 4 <= ids_size; i += 4) {
      int16x4_t weights4 = vld1_s16(&k[i]);

      // Row 0
      uint8x8x3_t rgb0 = vld3_u8(lineIn0_min + stride * i);
      int16x4_t r0_16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[0])));
      int16x4_t g0_16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[1])));
      int16x4_t b0_16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb0.val[2])));
      acc0_r = vmlal_s16(acc0_r, r0_16, weights4);
      acc0_g = vmlal_s16(acc0_g, g0_16, weights4);
      acc0_b = vmlal_s16(acc0_b, b0_16, weights4);

      // Row 1
      uint8x8x3_t rgb1 = vld3_u8(lineIn1_min + stride * i);
      int16x4_t r1_16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[0])));
      int16x4_t g1_16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[1])));
      int16x4_t b1_16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb1.val[2])));
      acc1_r = vmlal_s16(acc1_r, r1_16, weights4);
      acc1_g = vmlal_s16(acc1_g, g1_16, weights4);
      acc1_b = vmlal_s16(acc1_b, b1_16, weights4);
    }

    // Horizontal sum
    int32_t sum0_r = vaddvq_s32(acc0_r) + initial_val;
    int32_t sum0_g = vaddvq_s32(acc0_g) + initial_val;
    int32_t sum0_b = vaddvq_s32(acc0_b) + initial_val;
    int32_t sum1_r = vaddvq_s32(acc1_r) + initial_val;
    int32_t sum1_g = vaddvq_s32(acc1_g) + initial_val;
    int32_t sum1_b = vaddvq_s32(acc1_b) + initial_val;

    // Scalar cleanup for remaining pixels
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p0 = lineIn0_min + stride * i;
      const uint8_t* p1 = lineIn1_min + stride * i;
      sum0_r += w * p0[0];
      sum0_g += w * p0[1];
      sum0_b += w * p0[2];
      sum1_r += w * p1[0];
      sum1_g += w * p1[1];
      sum1_b += w * p1[2];
    }

    // Shift, clamp, and store row 0
    uint8_t* out0 = lineOut0 + stride * out_x;
    out0[0] = static_cast<uint8_t>(std::clamp(sum0_r >> coefs_precision, 0, 255));
    out0[1] = static_cast<uint8_t>(std::clamp(sum0_g >> coefs_precision, 0, 255));
    out0[2] = static_cast<uint8_t>(std::clamp(sum0_b >> coefs_precision, 0, 255));

    // Shift, clamp, and store row 1
    uint8_t* out1 = lineOut1 + stride * out_x;
    out1[0] = static_cast<uint8_t>(std::clamp(sum1_r >> coefs_precision, 0, 255));
    out1[1] = static_cast<uint8_t>(std::clamp(sum1_g >> coefs_precision, 0, 255));
    out1[2] = static_cast<uint8_t>(std::clamp(sum1_b >> coefs_precision, 0, 255));
  }
}

// Horizontal convolution for a single output row - optimized for RGB
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

    // Separate accumulators for R, G, B channels
    int32x4_t acc_r = vdupq_n_s32(0);
    int32x4_t acc_g = vdupq_n_s32(0);
    int32x4_t acc_b = vdupq_n_s32(0);

    int64_t i = 0;

    // Process 8 pixels at a time using vld3 to deinterleave RGB
    for (; i + 8 <= ids_size; i += 8) {
      // Load and deinterleave 8 RGB pixels (24 bytes)
      // This separates R, G, B into separate vectors
      uint8x8x3_t rgb = vld3_u8(lineIn_min + stride * i);

      // Load 8 weights
      int16x8_t weights = vld1q_s16(&k[i]);

      // Widen each channel from uint8 to int16
      int16x8_t r16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[0]));
      int16x8_t g16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[1]));
      int16x8_t b16 = vreinterpretq_s16_u16(vmovl_u8(rgb.val[2]));

      // Multiply and widen to 32-bit, accumulate low and high halves
      acc_r = vmlal_s16(acc_r, vget_low_s16(r16), vget_low_s16(weights));
      acc_r = vmlal_s16(acc_r, vget_high_s16(r16), vget_high_s16(weights));
      acc_g = vmlal_s16(acc_g, vget_low_s16(g16), vget_low_s16(weights));
      acc_g = vmlal_s16(acc_g, vget_high_s16(g16), vget_high_s16(weights));
      acc_b = vmlal_s16(acc_b, vget_low_s16(b16), vget_low_s16(weights));
      acc_b = vmlal_s16(acc_b, vget_high_s16(b16), vget_high_s16(weights));
    }

    // Process 4 pixels at a time
    for (; i + 4 <= ids_size; i += 4) {
      // Load 4 RGB pixels (12 bytes) - use vld3 with lane variant
      uint8x8x3_t rgb = vld3_u8(lineIn_min + stride * i);

      // Load 4 weights (zero-extend to 8)
      int16x4_t weights4 = vld1_s16(&k[i]);

      // Widen pixels to 16-bit (only use low 4 values)
      int16x4_t r16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[0])));
      int16x4_t g16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[1])));
      int16x4_t b16 = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(rgb.val[2])));

      // Multiply and widen to 32-bit, then accumulate
      acc_r = vmlal_s16(acc_r, r16, weights4);
      acc_g = vmlal_s16(acc_g, g16, weights4);
      acc_b = vmlal_s16(acc_b, b16, weights4);
    }

    // Horizontal sum of accumulators
    int32_t sum_r = vaddvq_s32(acc_r) + initial_val;
    int32_t sum_g = vaddvq_s32(acc_g) + initial_val;
    int32_t sum_b = vaddvq_s32(acc_b) + initial_val;

    // Handle remaining pixels with scalar code
    for (; i < ids_size; i++) {
      int16_t w = k[i];
      const uint8_t* p = lineIn_min + stride * i;
      sum_r += w * p[0];
      sum_g += w * p[1];
      sum_b += w * p[2];
    }

    // Shift, clamp, and store
    uint8_t* out = lineOut + stride * out_x;
    out[0] = static_cast<uint8_t>(std::clamp(sum_r >> coefs_precision, 0, 255));
    out[1] = static_cast<uint8_t>(std::clamp(sum_g >> coefs_precision, 0, 255));
    out[2] = static_cast<uint8_t>(std::clamp(sum_b >> coefs_precision, 0, 255));
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

      NeonResampleHorizontalConvolution8u4x(
          buf_row0, buf_row1, buf_row2, buf_row3,
          xout,
          in_row0, in_row1, in_row2, in_row3,
          xin,
          horiz_weights.xmin.data(),
          horiz_weights.xsize.data(),
          horiz_weights.weights.data(),
          horiz_weights.kmax,
          horiz_weights.precision,
          num_channels);
    }
    // Handle remaining 2 rows
    for (; y + 2 <= yin; y += 2) {
      const uint8_t* in_row0 = input_batch + y * in_row_stride;
      const uint8_t* in_row1 = input_batch + (y + 1) * in_row_stride;
      uint8_t* buf_row0 = buffer_ptr + y * buffer_row_stride;
      uint8_t* buf_row1 = buffer_ptr + (y + 1) * buffer_row_stride;

      NeonResampleHorizontalConvolution8u2x(
          buf_row0, buf_row1,
          xout,
          in_row0, in_row1,
          xin,
          horiz_weights.xmin.data(),
          horiz_weights.xsize.data(),
          horiz_weights.weights.data(),
          horiz_weights.kmax,
          horiz_weights.precision,
          num_channels);
    }
    // Handle remaining 1 row
    for (; y < yin; y++) {
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
