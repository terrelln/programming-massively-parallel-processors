from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="chapter6",
    ext_modules=[
        CUDAExtension(
            "chapter6",
            [
                "chapter6_ext.cpp",
                "chapter6.cu",
            ],
            libraries=["cuda"],
            define_macros=[("GLOG_USE_GLOG_EXPORT", None)]
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)