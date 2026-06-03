import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

def read_version():
    """Read the version number from the VERSION file."""
    version_file = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("version ="):
                version = line.split("=")[1].strip().strip('"')
                return version
    raise ValueError("Version not found in pyproject.toml")

def get_extensions():
    """Build C++ and CUDA extensions for the geocutool package."""
    
    csrc_dir = os.path.join("geocutool", "csrc")
    main_file = [os.path.join(csrc_dir, "pybind.cpp")]
    source_cuda = glob.glob(os.path.join(csrc_dir, "**", "*.cu"), recursive=True)
    source_cpp = [
        f for f in glob.glob(os.path.join(csrc_dir, "**", "*.cpp"), recursive=True)
        if os.path.basename(f) != "pybind.cpp"
    ]
    
    sources = main_file + source_cpp
    extension = CppExtension

    define_macros = []
    extra_compile_args = {}
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"
    
    # Try to find CUDA_HOME: first check standard location, then CONDA_PREFIX
    cuda_home = CUDA_HOME
    if cuda_available and cuda_home is None:
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            print(f"Checking for CUDA in CONDA_PREFIX: {conda_prefix}")
            # Check if CUDA headers exist in the conda environment
            cuda_include = os.path.join(conda_prefix, "include", "cuda.h")
            cuda_lib = os.path.join(conda_prefix, "lib", "libcudart.so")
            if os.path.isfile(cuda_include) or os.path.isfile(cuda_lib):
                cuda_home = conda_prefix
                print(f"Found CUDA in conda environment: {cuda_home}")
    
    if (cuda_available and cuda_home is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            major, minor = torch.cuda.get_device_capability()
            nvcc_flags = [f"-arch=sm_{major}{minor}"]
            # nvcc_flags = ["-O3"]
            # nvcc_flags += ['-DTORCH_INDUCTOR_CPP_WRAPPER']
        else:
            nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args = {
            "cxx": ["-O3"],
            "nvcc": nvcc_flags,
        }
        print(f"Building CUDA extension with CUDA_HOME: {cuda_home}")
    else:
        print("Building CPU-only extension (CUDA not available)")

    sources = [s for s in sources]
    include_dirs = [os.path.join("geocutool", "csrc")]
    print("sources:", sources)
    print("include_dirs:", include_dirs)

    ext_modules = [
        extension(
            "geocutool._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="geocutool",
    version=read_version(),
    author="",
    author_email="",
    keywords="marching cubes differentiable",
    description="Differentiable Marching Cubes Package",
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    license="MIT",
    packages=find_packages(exclude=["examples", "tests"]),
    python_requires=">=3.8",
    install_requires=["torch"],
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
    zip_safe=False
)