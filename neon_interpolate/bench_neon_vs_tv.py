#!/usr/bin/env python
"""Benchmark: neon_interpolate vs torchvision resize (bilinear, antialias=True)."""

import torch
from time import perf_counter_ns
from torchvision.transforms.v2.functional import resize, InterpolationMode
import neon_interpolate


def bench(f, *args, num_exp=15, warmup=2):
    for _ in range(warmup):
        f(*args)
    times = []
    for _ in range(num_exp):
        t0 = perf_counter_ns()
        f(*args)
        times.append(perf_counter_ns() - t0)
    return torch.tensor(times).float().median().item() * 1e-6  # ms


torch.set_num_threads(1)

resolutions = [
    ("64x64", (1, 3, 64, 64)),
    ("320p", (1, 3, 320, 480)),
    ("720p", (1, 3, 720, 1280)),
    ("4K", (1, 3, 2160, 3840)),
]
scale_factors = [0.25, 0.5, 0.75, 1.5, 2.0, 4.0]

backends = [
    (
        "tv",
        lambda x, t: resize(
            x, t, interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
    ),
    ("neon", lambda x, t: neon_interpolate.neon_upsample_bilinear2d_aa(x, t)),
    (
        "tiled_neon",
        lambda x, t: neon_interpolate.tiled_neon_upsample_bilinear2d_aa(x, t),
    ),
    ("ring_neon", lambda x, t: neon_interpolate.ring_neon_upsample_bilinear2d_aa(x, t)),
]

layouts = [
    ("cf", lambda x: x),
    ("cl", lambda x: x.contiguous(memory_format=torch.channels_last)),
]

# results[(name, sf)][(backend, layout)] = time_ms
results = {}
for name, shape in resolutions:
    torch.manual_seed(42)
    x_orig = torch.randint(0, 256, shape, dtype=torch.uint8)
    for sf in scale_factors:
        target = [max(1, int(shape[2] * sf)), max(1, int(shape[3] * sf))]
        key = (name, sf)
        results[key] = {}

        parts = []
        for bl, bfn in backends:
            for ll, lfn in layouts:
                x = lfn(x_orig)
                t = bench(lambda _x=x, _t=target: bfn(_x, _t))
                results[key][(bl, ll)] = t
                parts.append(f"{bl}_{ll}={t:.2f}ms")

        print(f"{name} x{sf}: " + "  ".join(parts), flush=True)

col = 10
header = f"{'':>6}" + "".join(f"{'x' + str(sf):>{col}}" for sf in scale_factors)

# For each non-tv backend+layout, show speedup vs tv with same layout
for bl, _ in backends[1:]:
    for ll, _ in layouts:
        label = f"Speedup (tv_{ll} / {bl}_{ll})"
        print(f"\n{label:^{6 + col * len(scale_factors)}}")
        print(header)
        print("-" * len(header))
        for name, _ in resolutions:
            row = f"{name:>6}"
            for sf in scale_factors:
                r = results[(name, sf)]
                row += f"{r[('tv', ll)] / r[(bl, ll)]:>{col}.2f}x"
            print(row)
