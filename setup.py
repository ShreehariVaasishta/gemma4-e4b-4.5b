from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Notice there is no name, version, or dependencies here!
# All of that is handled by pyproject.toml now.
setup(
    ext_modules=[
        CUDAExtension(
            name="gemma4_kernels",
            sources=glob("gemma4/kernels/*.cpp") + glob("gemma4/kernels/*.cu"),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
