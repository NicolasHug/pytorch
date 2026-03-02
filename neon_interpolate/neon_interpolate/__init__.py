from . import _C
from . import _C_tiled
from . import _C_tiled_neon
from . import _C_ring
from . import _C_ring_neon


def neon_upsample_bilinear2d_aa(input, output_size):
    """NEON-accelerated bilinear upsampling with antialiasing."""
    return _C.ops.neon_interpolate.upsample_bilinear2d_aa(input, output_size)


def tiled_upsample_bilinear2d_aa(input, output_size):
    """Tiled separable bilinear resize with antialiasing (scalar C++)."""
    return _C_tiled.ops.tiled_interpolate.upsample_bilinear2d_aa(input, output_size)


def tiled_neon_upsample_bilinear2d_aa(input, output_size):
    """Tiled separable bilinear resize with antialiasing (NEON)."""
    return _C_tiled_neon.ops.tiled_neon_interpolate.upsample_bilinear2d_aa(input, output_size)


def ring_upsample_bilinear2d_aa(input, output_size):
    """Ring-buffer separable bilinear resize with antialiasing (scalar C++)."""
    return _C_ring.ops.ring_interpolate.upsample_bilinear2d_aa(input, output_size)


def ring_neon_upsample_bilinear2d_aa(input, output_size):
    """Ring-buffer separable bilinear resize with antialiasing (NEON)."""
    return _C_ring_neon.ops.ring_neon_interpolate.upsample_bilinear2d_aa(input, output_size)
