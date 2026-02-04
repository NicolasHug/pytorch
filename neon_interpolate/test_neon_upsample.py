"""
Test and benchmark for NEON bilinear interpolation with antialias.

Compares the NEON implementation against PyTorch's uint8 reference implementation.
"""

import torch
import time
import sys

sys.path.insert(0, ".")
import neon_interpolate


def test_basic():
    """Basic correctness tests."""
    print("=== Basic correctness tests ===\n")

    test_cases = [
        ("Downscale 64x64 -> 32x32", (1, 3, 64, 64), (32, 32)),
        ("Upscale 32x32 -> 64x64", (1, 3, 32, 32), (64, 64)),
        ("Asymmetric 64x32 -> 32x64", (1, 3, 64, 32), (32, 64)),
        ("Small 4x4 -> 8x8", (1, 3, 4, 4), (8, 8)),
        ("Large downscale 256x256 -> 32x32", (1, 3, 256, 256), (32, 32)),
        ("Non-power-of-2 100x75 -> 50x40", (1, 3, 100, 75), (50, 40)),
        ("Batch size 4", (4, 3, 64, 64), (32, 32)),
    ]

    all_passed = True
    for name, input_shape, output_size in test_cases:
        torch.manual_seed(42)
        input_tensor = torch.randint(0, 256, input_shape, dtype=torch.uint8)

        # NEON implementation
        output_neon = neon_interpolate.neon_upsample_bilinear2d_aa(
            input_tensor, list(output_size)
        )

        # Reference uint8 implementation
        output_ref = torch.nn.functional.interpolate(
            input_tensor, size=output_size, mode="bilinear", antialias=True
        )

        diff = (output_neon.float() - output_ref.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        status = "PASS" if max_diff <= 1 else "FAIL"
        if max_diff > 1:
            all_passed = False

        print(f"{name}:")
        print(f"  Max diff: {max_diff}, Mean diff: {mean_diff:.4f} [{status}]")

    print()
    return all_passed


def test_memory_formats():
    """Test different memory formats."""
    print("=== Memory format tests ===\n")

    torch.manual_seed(42)
    input_cont = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    input_cl = input_cont.contiguous(memory_format=torch.channels_last)

    # Test contiguous input
    output_cont = neon_interpolate.neon_upsample_bilinear2d_aa(input_cont, [32, 32])
    ref_cont = torch.nn.functional.interpolate(
        input_cont, size=(32, 32), mode="bilinear", antialias=True
    )
    diff_cont = (output_cont.float() - ref_cont.float()).abs().max().item()

    # Test channels_last input
    output_cl = neon_interpolate.neon_upsample_bilinear2d_aa(input_cl, [32, 32])
    ref_cl = torch.nn.functional.interpolate(
        input_cl, size=(32, 32), mode="bilinear", antialias=True
    )
    diff_cl = (output_cl.float() - ref_cl.float()).abs().max().item()

    print(f"Contiguous input:")
    print(
        f"  Input is channels_last: {input_cont.is_contiguous(memory_format=torch.channels_last)}"
    )
    print(f"  Output is contiguous: {output_cont.is_contiguous()}")
    print(f"  Max diff: {diff_cont}")

    print(f"\nChannels_last input:")
    print(
        f"  Input is channels_last: {input_cl.is_contiguous(memory_format=torch.channels_last)}"
    )
    print(
        f"  Output is channels_last: {output_cl.is_contiguous(memory_format=torch.channels_last)}"
    )
    print(f"  Max diff: {diff_cl}")

    print()
    return diff_cont <= 1 and diff_cl <= 1


def benchmark():
    """Benchmark NEON vs reference uint8 implementation."""
    print("=== Benchmark (single-threaded) ===\n")

    # Force single-threaded execution for fair comparison
    torch.set_num_threads(1)
    print(f"torch.get_num_threads() = {torch.get_num_threads()}\n")

    configs = [
        # Original benchmarks
        ("224x224 -> 112x112 (downscale 2x)", (4, 3, 224, 224), (112, 112)),
        ("224x224 -> 448x448 (upscale 2x)", (4, 3, 224, 224), (448, 448)),
        ("512x512 -> 256x256 (downscale 2x)", (1, 3, 512, 512), (256, 256)),
        ("64x64 -> 224x224 (upscale ~3.5x)", (4, 3, 64, 64), (224, 224)),
        # 720p (1280x720) benchmarks
        ("720p -> 360p (downscale 2x)", (1, 3, 720, 1280), (360, 640)),
        ("720p -> 480p (downscale ~1.5x)", (1, 3, 720, 1280), (480, 854)),
        ("720p -> 256x256 (downscale)", (1, 3, 720, 1280), (256, 256)),
        # 4K (3840x2160) benchmarks
        ("4K -> 1080p (downscale 2x)", (1, 3, 2160, 3840), (1080, 1920)),
        ("4K -> 720p (downscale 3x)", (1, 3, 2160, 3840), (720, 1280)),
        ("4K -> 256x256 (downscale)", (1, 3, 2160, 3840), (256, 256)),
    ]

    n_warmup = 3
    n_iters = 10  # Fewer iterations for large images

    for name, input_shape, output_size in configs:
        torch.manual_seed(42)
        input_tensor = torch.randint(0, 256, input_shape, dtype=torch.uint8)

        # Warmup
        for _ in range(n_warmup):
            _ = neon_interpolate.neon_upsample_bilinear2d_aa(
                input_tensor, list(output_size)
            )
            _ = torch.nn.functional.interpolate(
                input_tensor, size=output_size, mode="bilinear", antialias=True
            )

        # Benchmark NEON
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = neon_interpolate.neon_upsample_bilinear2d_aa(
                input_tensor, list(output_size)
            )
        neon_time = (time.perf_counter() - start) / n_iters * 1000

        # Benchmark reference uint8
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = torch.nn.functional.interpolate(
                input_tensor, size=output_size, mode="bilinear", antialias=True
            )
        ref_time = (time.perf_counter() - start) / n_iters * 1000

        speedup = ref_time / neon_time
        print(f"{name}:")
        print(f"  Input shape: {input_shape}")
        print(f"  NEON:      {neon_time:.3f} ms")
        print(f"  Reference: {ref_time:.3f} ms")
        print(f"  Speedup:   {speedup:.2f}x")
        print()


def main():
    print("=" * 60)
    print("NEON Bilinear Interpolation with Antialias - Test Suite")
    print("=" * 60)
    print()

    # Run tests
    basic_passed = test_basic()
    format_passed = test_memory_formats()

    # Run benchmark
    benchmark()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Basic tests: {'PASSED' if basic_passed else 'FAILED'}")
    print(f"Memory format tests: {'PASSED' if format_passed else 'FAILED'}")

    if basic_passed and format_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
