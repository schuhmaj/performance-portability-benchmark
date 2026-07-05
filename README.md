# Performance Portability Benchmarking

This repository contains several examples of the same algorithm,
but implemented employing different libraries, frameworks, programming paradigms
and compilers used for performance portability.
Some frameworks are not portable to every platform.


<p align="center">
  <img src="results/all_scatter.png" width="90%">
  <br>
  <em>
      Normalized Halstead Effort (Complexity) vs. Application Efficiency (Runtime)<br>
      performed on an RTX 2080 with LLVM 20.1.8/ NVHPC 25.9
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

## Polyhedral Gravity Model

Please find the code for the polyhedral gravity implementations here: https://github.com/rho2/polyhedral-gravity-parallel

## Build

### Build Options

To execute all examples, one requires at least
two build folders since not every technology runs with
every compiler - even tough some of them can be compiled
with both options.
By default **all** CMake Targets are disabled. One have
to explicit enable one technology:

| Option Name                      | Description                                     | Toolchain              |
|----------------------------------|-------------------------------------------------|------------------------|
| PPB_LOGGING_LEVEL                | Logging Level                                   | N/A                    |
| PPB_ENABLE_OnlyKernelRuntime     | Only Kernel Runtime (otherwise Wall Clock Time) | N/A                    |
| PPB_ENABLE_Kokkos                | Kokkos                                          | LLVM, NVHPC            |
| PPB_ENABLE_Raja                  | RAJA                                            | LLVM, NVHPC            |
| PPB_ENABLE_AdaptiveCpp           | AdaptiveCpp (needs Boost)                       | LLVM, NVHPC            |
| PPB_ENABLE_Vulkan                | Vulkan-Kompute                                  | LLVM + Vulkan          |
| PPB_ENABLE_OpenACC               | OpenACC                                         | NVHPC or LLLVM (clacc) |
| PPB_ENABLE_OpenMP                | OpenMP                                          | LLVM                   |
| PPB_ENABLE_OpenCL                | OpenCL                                          | LLVM, NVHPC            |
| PPB_ENABLE_Boost                 | Boost::Compute (needs OpenCL)                   | LLVM, NVHPC            |
| PPB_ENABLE_CUDA                  | Cuda                                            | LLVM, NVHPC            |
| PPB_ENABLE_Alpaka                | Alpaka                                          | LLVM, NVHPC            |
| PPB_ENABLE_Metal                 | Metal (Apple)                                   | Apple Clang (macOS)    |
| PPB_ENABLE_VectorAddition        | VectorAddition                                  | N/A                    |
| PPB_ENABLE_MatrixMultiplication  | MatrixMultiplication                            | N/A                    |
| PPB_ENABLE_NBodySimulation       | NBodySimulation                                 | N/A                    |
| PPB_ENABLE_Testing               | Testing                                         | N/A                    |

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