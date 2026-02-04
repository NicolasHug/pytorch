from . import _C


def neon_upsample_bilinear2d_aa(input, output_size):
    """NEON-accelerated bilinear upsampling with antialiasing."""
    return _C.ops.neon_interpolate.upsample_bilinear2d_aa(input, output_size)
