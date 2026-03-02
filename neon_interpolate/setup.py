from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="neon_interpolate",
    ext_modules=[
        cpp_extension.CppExtension(
            "neon_interpolate._C",
            ["csrc/neon_upsample.cpp"],
            extra_compile_args={"cxx": ["-O3"]},
        ),
        cpp_extension.CppExtension(
            "neon_interpolate._C_sve2",
            ["csrc/sve2_upsample.cpp"],
            extra_compile_args={"cxx": ["-O3", "-march=native"]},
        ),
        cpp_extension.CppExtension(
            "neon_interpolate._C_tiled",
            ["csrc/tiled.cpp"],
            extra_compile_args={"cxx": ["-O3"]},
        ),
        cpp_extension.CppExtension(
            "neon_interpolate._C_tiled_neon",
            ["csrc/tiled_neon.cpp"],
            extra_compile_args={"cxx": ["-O3"]},
        ),
        cpp_extension.CppExtension(
            "neon_interpolate._C_ring",
            ["csrc/ring_buffer.cpp"],
            extra_compile_args={"cxx": ["-O3"]},
        ),
        cpp_extension.CppExtension(
            "neon_interpolate._C_ring_neon",
            ["csrc/ring_buffer_neon.cpp"],
            extra_compile_args={"cxx": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=["neon_interpolate"],
)
