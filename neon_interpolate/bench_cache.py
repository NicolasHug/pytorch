"""Run a single resize backend in a loop for perf-stat measurement.

Usage:
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        python bench_cache.py tiled
    perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        python bench_cache.py tv
"""

import sys
import torch

torch.set_num_threads(1)

backend = sys.argv[1]

x = torch.randint(0, 256, (1, 3, 2160, 3840), dtype=torch.uint8)
target = [1080, 1920]
n_iters = 20

if backend == "tiled":
    import neon_interpolate
    import neon_interpolate._C_tiled
    f = lambda: neon_interpolate.tiled_upsample_bilinear2d_aa(x, target)
elif backend == "tv":
    from torchvision.transforms.v2.functional import resize, InterpolationMode
    f = lambda: resize(x, target, interpolation=InterpolationMode.BILINEAR, antialias=True)
elif backend == "neon":
    import neon_interpolate
    f = lambda: neon_interpolate.neon_upsample_bilinear2d_aa(x, target)
elif backend == "tiled_neon":
    import neon_interpolate
    f = lambda: neon_interpolate.tiled_neon_upsample_bilinear2d_aa(x, target)
else:
    print(f"Unknown backend: {backend}. Use 'tiled', 'tiled_neon', 'tv', or 'neon'.")
    sys.exit(1)

# Warmup
for _ in range(3):
    f()

for _ in range(n_iters):
    f()
