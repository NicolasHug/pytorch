#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <algorithm>

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

void scalar_horizontal_conv_uint8(
    uint8_t* out,
    const uint8_t* in,
    int64_t W_out,
    const int64_t* xmin,
    const int64_t* xsize,
    const int16_t* weights,
    int kmax,
    unsigned int precision,
    int64_t C) {
  const int32_t round = 1 << (precision - 1);
  for (int64_t ox = 0; ox < W_out; ox++) {
    const uint8_t* in_ptr = in + xmin[ox];
    const int16_t* wt = &weights[ox * kmax];
    int64_t sz = xsize[ox];
    for (int64_t c = 0; c < C; c++) {
      int32_t acc = round;
      for (int64_t j = 0; j < sz; j++) {
        acc += static_cast<int32_t>(in_ptr[j * C + c]) * wt[j];
      }
      out[ox * C + c] = static_cast<uint8_t>(
          std::clamp(acc >> static_cast<int32_t>(precision), 0, 255));
    }
  }
}

void ring_vertical_conv_uint8(
    uint8_t* out,
    const uint8_t* ring_buf,
    int64_t row_bytes,
    int64_t ring_capacity,
    int64_t first_row,
    int64_t num_rows,
    const int16_t* weights,
    unsigned int precision) {
  const int32_t round = 1 << (precision - 1);

  // Precompute row pointers so the j loop has simple contiguous access,
  // enabling auto-vectorization (the modulo made it opaque to the compiler).
  const uint8_t* row_ptrs[32];
  for (int64_t i = 0; i < num_rows; i++) {
    row_ptrs[i] = ring_buf + ((first_row + i) % ring_capacity) * row_bytes;
  }

  for (int64_t j = 0; j < row_bytes; j++) {
    int32_t acc = round;
    for (int64_t i = 0; i < num_rows; i++) {
      acc += static_cast<int32_t>(row_ptrs[i][j]) * weights[i];
    }
    out[j] = static_cast<uint8_t>(
        std::clamp(acc >> static_cast<int32_t>(precision), 0, 255));
  }
}

void ring_buffer_bilinear_uint8(const at::Tensor& input, const at::Tensor& output) {
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

  // Ring buffer: kmax rows of W_out * C bytes.
  // For 4K->1080p this is ~6 * 5760 = ~34KB, fitting in L1 cache.
  auto ring_buf_tensor = at::empty({ring_capacity * row_bytes}, input.options());
  uint8_t* ring_buf = ring_buf_tensor.data_ptr<uint8_t>();

  const uint8_t* input_ptr = input.const_data_ptr<uint8_t>();
  uint8_t* output_ptr = output.data_ptr<uint8_t>();
  const int64_t in_row_stride = W_in * C;

  int64_t last_computed_row = -1;

  for (int64_t yy = 0; yy < H_out; yy++) {
    int64_t first_needed = vert_w.xmin[yy] / row_bytes;
    int64_t num_needed = vert_w.xsize[yy];
    int64_t last_needed = first_needed + num_needed - 1;

    // Horizontally scale only the new input rows into the ring buffer
    for (int64_t row = last_computed_row + 1; row <= last_needed; row++) {
      scalar_horizontal_conv_uint8(
          ring_buf + (row % ring_capacity) * row_bytes,
          input_ptr + row * in_row_stride,
          W_out,
          horiz_w.xmin.data(), horiz_w.xsize.data(),
          horiz_w.weights.data(), horiz_w.kmax,
          horiz_w.precision, C);
    }
    last_computed_row = std::max(last_computed_row, last_needed);

    // Vertical convolution reading from ring buffer
    ring_vertical_conv_uint8(
        output_ptr + yy * row_bytes,
        ring_buf,
        row_bytes,
        ring_capacity,
        first_needed,
        num_needed,
        &vert_w.weights[yy * vert_w.kmax],
        vert_w.precision);
  }
}

} // anonymous namespace

at::Tensor upsample_bilinear2d_aa_ring(
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

  ring_buffer_bilinear_uint8(input_cl, output);

  if (!input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return output.contiguous();
  }
  return output;
}

TORCH_LIBRARY(ring_interpolate, m) {
  m.def("upsample_bilinear2d_aa(Tensor input, int[] output_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(ring_interpolate, CPU, m) {
  m.impl("upsample_bilinear2d_aa", &upsample_bilinear2d_aa_ring);
}

PYBIND11_MODULE(_C_ring, m) {
  m.attr("ops") = py::module::import("torch").attr("ops");
}
