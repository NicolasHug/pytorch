#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/extension.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

at::Tensor upsample_bilinear2d_aa_neon(
    const at::Tensor& input,
    at::IntArrayRef output_size) {

  TORCH_CHECK(input.dim() == 4, "Expected 4D input");
  TORCH_CHECK(output_size.size() == 2, "Expected 2D output size");
  TORCH_CHECK(input.device().is_cpu(), "Expected CPU tensor");

  auto batch = input.size(0);
  auto channels = input.size(1);
  auto out_h = output_size[0];
  auto out_w = output_size[1];

  auto output = at::empty({batch, channels, out_h, out_w}, input.options());

#if defined(__aarch64__)
  // Test NEON: simple vector add to confirm NEON works
  float32x4_t a = vdupq_n_f32(1.5f);
  float32x4_t b = vdupq_n_f32(2.5f);
  float32x4_t c = vaddq_f32(a, b);
  float result[4];
  vst1q_f32(result, c);
  printf("NEON test: 1.5 + 2.5 = %f\n", result[0]);

  // TODO: Actual NEON implementation
  output.zero_();
#else
  TORCH_CHECK(false, "NEON not available on this platform");
#endif

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
