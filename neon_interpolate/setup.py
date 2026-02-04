from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="neon_interpolate",
    ext_modules=[
        cpp_extension.CppExtension(
            "neon_interpolate._C",
            ["csrc/neon_upsample.cpp"],
            extra_compile_args={"cxx": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=["neon_interpolate"],
)
