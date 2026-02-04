/*
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is

    Copyright © 2010-2022 by Alex Clark and contributors

Like PIL, Pillow is licensed under the open source HPND License
*/

// This file provides NEON-optimized kernels for uint8 bilinear/bicubic
// interpolation with antialiasing on aarch64, modeled after
// UpSampleKernelAVXAntialias.h for AVX2.
//
// Currently this is a placeholder. The actual NEON intrinsics implementation
// will be added in a future PR. The stub implementation (which falls back to
// the generic separable implementation) is defined directly in
// UpSampleKernel.cpp.

#pragma once
#if defined(__aarch64__) && !defined(C10_MOBILE)

#include <ATen/core/Tensor.h>
#include <arm_neon.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace {

// Helper functions for pack/unpack and NEON convolutions will be added here.
// The main entry point will be:
//
// template <typename scale_type, class F>
// void upsample_neon_bilinear_bicubic_uint8(
//     const at::Tensor& input,
//     const at::Tensor& output,
//     bool align_corners,
//     const scale_type& scales,
//     bool antialias);

} // anonymous namespace

#endif // defined(__aarch64__) && !defined(C10_MOBILE)
