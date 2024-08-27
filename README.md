# Performance Portability Benchmarking

This repository contains several examples of the same algorithm,
but implemented employing different libraries, frameworks, programming paradigms
and compilers used for performance portability.
Some frameworks are not portable to every platform.

## Build

### Example Set-Up for CUDA/ Nvidia HPC SDK

Execute in the repository root the following commands, i.e., adapt them
according to your needs:

```bash
mkdir build && cd build

cmake .. -G Ninja \
  -DPPB_ENABLE_AdaptiveCpp=ON \        # Enable AdaptiveCpp/OpenSYCL
  -DACPP_TARGETS="cuda:sm_86" \        # Configure the Target of AdaptiveCpp/OpenSYCL
  -DPPB_ENABLE_OpenCL=ON \             # Enable BoostCompute and OpenCL
  -DPPB_ENABLE_OpenACC=ON \            # Enable OpenACC
  -DPPB_ENABLE_OpenMP=ON \             # Enable OpenMP
  -DCMAKE_C_COMPILER=nvc \             # Set the C compiler
  -DCMAKE_CXX_COMPILER=nvc++ \         # Set the C++ compiler
  -DCMAKE_CUDA_HOST_COMPILER=nvc++ \   # Set the C++ compiler to which CUDA Compiler offloads
  -DCMAKE_CUDA_COMPILER=nvcc \         # Set CUDA compiler
  -DKokkos_ENABLE_CUDA=ON \            # Enable Kokkos CUDA backend
  -DKokkos_ARCH_AMPERE86=ON            # Set the Target Architecture of Kokkos' Device Backend

cmake --build .                       # to build everything available
cmake --build . --target vec_acc     # or whatever target you require
```

