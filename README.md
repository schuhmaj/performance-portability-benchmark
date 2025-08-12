# Performance Portability Benchmarking

This repository contains several examples of the same algorithm,
but implemented employing different libraries, frameworks, programming paradigms
and compilers used for performance portability.
Some frameworks are not portable to every platform.


<p align="center">
  <img src="results/vectorAddition/2025-06-04_19-41_Benchmark_Result.png" width="90%">
  <br>
  <em>
    Runtime of a Simple Vector Addition on the RTX 3080 using Different Pardigms
  </em>
</p>

## Requirements

In order to properly work with **all** implementation, please ensure the presence
of the following two options:

- Option 1: **NVHPC-based**
  - [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)
- Option 2: **LLVM-based**
  - [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  - [LLVM Compiler and Libraries](https://github.com/llvm/llvm-project), alternatively [Clang with OpenACC support](https://github.com/llvm-doe-org/llvm-project/tree/clacc/main).
    Compile your own LLVM by downloading the [latest release](https://github.com/llvm/llvm-project/releases):
  ```bash
  tar -xvf llvmorg-<version>.tar.gz
  cd llvm-project-llvmorg-<version>/llvm
  cp <this-repo>/tools/llvm_CMakePresets.json CMakePresets.json
  # Before executing the next line, modify CMAKE_INSTALL_PREFIX and CMAKE_INSTALL_RPATH in the Preset accordingly! 
  cmake --workflow --preset default 
  ```
- Option 3 **Intel oneAPI Based**
  - Install [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
  - Install the SYCL Extension from Codeplay for [Nvidia](https://developer.codeplay.com/products/oneapi/nvidia/download/) or [AMD](https://developer.codeplay.com/products/oneapi/amd/home/) GPUs
  - If Nvidia: Install [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) _before_ installing the SYCL Extension.
  - You can easily source the full environment by executing `source /opt/intel/oneapi/setvars.sh` after installation.

Requirements that might be necessary:

- OpenCL - This is usually installed as part of e.g. the Nvidia Tooling.
  If you require a dedicated installation,
  have a look at the [Portable Computing Language Project](https://github.com/pocl/pocl)
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) - This can be automatically set-up by CMake.
  However, due to higher compilation time/ effort, it is recommended to install it on your system/ in you userspace.


## Build

### Build Options

To execute all examples, one requires at least
two build folders since not every technology runs with
every compiler - even tough some of them can be compiled
with both options.
By default **all** CMake Targets are disabled. One have
to explicit enable one technology:


| Option Name              | Description                         | Toolchain   |
|--------------------------|-------------------------------------|-------------|
| PPB_ENABLE_OpenACC       | Enable OpenACC Target               | NVHPC       |
| PPB_ENABLE_OpenMP        | Enable OpenMP Target                | LLVM        |
| PPB_ENABLE_AdaptiveCpp   | Enable AdaptiveCPP/ SYCL Target     | LLVM        |
| PPB_ENABLE_OpenCL        | Enable OpenCL Targets (incl. Boost) | LLVM, NVHPC |
| PPB_ENABLE_Kokkos        | Enable Kokkos Target                | LLVM, NVHPC |
| PPB_ENABLE_Raja          | Enable Raja Target                  | LLVM, NVHPC |
| PPB_ENABLE_Cuda          | Enable Cuda and Thrust Targets      | LLVM, NVHPC |


### Typically Workflow

The recommended approach to build is using ninja, which is not necessarily required, but really a nice build tool :)

```bash
mkdir build && cd build
# Setup your environment by module load <compiler>, conda activate <environment>, etc.
cmake .. -G Ninja
# Select the options
ccmake ..
# Build all targets which are configured
cmake --build .
# Build one target, here: vec_acpp
cmake --build . --target vec_acpp
```

### SCCS Cluster

The SCCS Cluster offers four Nvidia RTX 3080 graphic cards.
To simplify the build process, the `tools` directory contains
two setup scripts to quickly get every target build.
Just execute in the repository root folder:

```bash
module load ninja-1.10.2 cmake-3.23.0 boost-1.69.0 llvm-14.0.0 cuda-11.6.0 intel-tbb-2020.3
./tools/sccs_cluster/setup_llvm.sh
module load module load ninja-1.10.2 cmake-3.23.0 boost-1.69.0 nvhpc-23.9
./tools/sccs_cluster/setup_nvhpc.sh
```

Further, [HOW-TO-MODULE](./tools/sccs_cluster/HOW-TO-MODULE.md)
contains instructions on how to simplify the local dependency management.

## Execution

The individual run-targets use Google Benchmark.
You find the available options by running

```bash
./vec_* --help
```

Batch Evaluation is simple using the provided Python script.
It requires `pandas`, `matplotlib`, `seaborn` and `loguru`.

Just execute in the repository root:

```bash
mkdir results && cd results
python ../scripts/plot_benchmark.py -r "vec_*" -p ..
```