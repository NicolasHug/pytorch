from . import _C
from . import _C_tiled
from . import _C_tiled_neon


def neon_upsample_bilinear2d_aa(input, output_size):
    """NEON-accelerated bilinear upsampling with antialiasing."""
    return _C.ops.neon_interpolate.upsample_bilinear2d_aa(input, output_size)


def tiled_upsample_bilinear2d_aa(input, output_size):
    """Tiled separable bilinear resize with antialiasing (scalar C++)."""
    return _C_tiled.ops.tiled_interpolate.upsample_bilinear2d_aa(input, output_size)


def tiled_neon_upsample_bilinear2d_aa(input, output_size):
    """Tiled separable bilinear resize with antialiasing (NEON)."""
    return _C_tiled_neon.ops.tiled_neon_interpolate.upsample_bilinear2d_aa(input, output_size)
