import torch
import pytest
import neon_interpolate


BACKENDS = [
    ("neon", neon_interpolate.neon_upsample_bilinear2d_aa),
    ("tiled", neon_interpolate.tiled_upsample_bilinear2d_aa),
    ("tiled_neon", neon_interpolate.tiled_neon_upsample_bilinear2d_aa),
    ("ring", neon_interpolate.ring_upsample_bilinear2d_aa),
    ("ring_neon", neon_interpolate.ring_neon_upsample_bilinear2d_aa),
]

RESOLUTIONS = [
    (64, 64),
    (320, 480),
    (720, 1280),
    (2160, 3840),
]

SCALE_FACTORS = [0.25, 0.5, 0.75, 1.5, 2.0, 4.0]

LAYOUTS = [
    ("cf", torch.contiguous_format),
    ("cl", torch.channels_last),
]


def reference(x, output_size):
    return torch.nn.functional.interpolate(
        x, size=output_size, mode="bilinear", antialias=True
    )


def _check(fn, input_tensor, output_size):
    out = fn(input_tensor, list(output_size))
    ref = reference(input_tensor, output_size)
    torch.testing.assert_close(out, ref, atol=1, rtol=0)


@pytest.mark.parametrize("scale", SCALE_FACTORS, ids=[f"x{s}" for s in SCALE_FACTORS])
@pytest.mark.parametrize("layout", LAYOUTS, ids=[l[0] for l in LAYOUTS])
@pytest.mark.parametrize("backend", BACKENDS, ids=[b[0] for b in BACKENDS])
def test_resolutions_and_scales(backend, layout, scale):
    name, fn = backend
    _, mem_fmt = layout
    for h, w in RESOLUTIONS:
        oh, ow = max(1, int(h * scale)), max(1, int(w * scale))
        torch.manual_seed(42)
        x = torch.randint(0, 256, (1, 3, h, w), dtype=torch.uint8)
        x = x.contiguous(memory_format=mem_fmt)
        _check(fn, x, (oh, ow))


@pytest.mark.parametrize("backend", BACKENDS, ids=[b[0] for b in BACKENDS])
def test_asymmetric(backend):
    _, fn = backend
    cases = [
        ((1, 3, 64, 32), (32, 64)),
        ((1, 3, 100, 75), (50, 40)),
        ((1, 3, 37, 53), (71, 29)),
    ]
    for shape, out_size in cases:
        torch.manual_seed(42)
        x = torch.randint(0, 256, shape, dtype=torch.uint8)
        _check(fn, x, out_size)


@pytest.mark.parametrize("backend", BACKENDS, ids=[b[0] for b in BACKENDS])
def test_small(backend):
    _, fn = backend
    cases = [
        ((1, 3, 2, 2), (4, 4)),
        ((1, 3, 4, 4), (8, 8)),
        ((1, 3, 4, 4), (2, 2)),
        ((1, 3, 3, 5), (7, 3)),
    ]
    for shape, out_size in cases:
        torch.manual_seed(42)
        x = torch.randint(0, 256, shape, dtype=torch.uint8)
        _check(fn, x, out_size)


@pytest.mark.parametrize("backend", BACKENDS, ids=[b[0] for b in BACKENDS])
def test_identity_scale(backend):
    _, fn = backend
    torch.manual_seed(42)
    x = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    _check(fn, x, (64, 64))


@pytest.mark.parametrize("backend", BACKENDS, ids=[b[0] for b in BACKENDS])
def test_large_downscale(backend):
    _, fn = backend
    torch.manual_seed(42)
    x = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)
    for out_size in [(16, 16), (32, 32)]:
        _check(fn, x, out_size)


def test_backends_agree():
    torch.manual_seed(42)
    x = torch.randint(0, 256, (1, 3, 100, 150), dtype=torch.uint8)
    for out_size in [(50, 75), (200, 300)]:
        ref = reference(x, out_size)
        for name, fn in BACKENDS:
            out = fn(x, list(out_size))
            torch.testing.assert_close(out, ref, atol=1, rtol=0)
