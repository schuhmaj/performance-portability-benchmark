#!/bin/bash
# =========================================================================
# SCCS CLUSTER SETUP
# =========================================================================
# Hardware: Nvidia RTX 3080
# Available Softwarestack: Cuda, Cuda-OpenCL, Nvidia HPC SDK
# =========================================================================
# USAGE INSTRUCTION
# =========================================================================
# This script sets up and builds a project using CMake.
# Usage: ./build_and_compile.sh [-p <build-dir>]
# -p <build-dir> : Specify the name of the build directory (default: build)
# =========================================================================

# Exit immediately if a command exits with a non-zero status
set -e
# Print commands and their arguments as they are executed
set -x

# Default build directory
BUILD_DIR="build"

# Parse command-line options
while getopts "p:" opt; do
  case "$opt" in
    p) BUILD_DIR="$OPTARG";;
    *) echo "Usage: $0 [-p <build-dir>]"; exit 1;;
  esac
done

echo "Loading required packages"
module load ninja-1.10.2 cmake-3.23.0 boost-1.69.0 nvhpc-23.9

echo "Creating build directory (${BUILD_DIR}) and navigating into it..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Running cmake configuration..."
cmake .. -G Ninja \
  -DPPB_ENABLE_AdaptiveCpp=ON \
  -DACPP_TARGETS="cuda:sm_86" \
  -DPPB_ENABLE_OpenCL=ON \
  -DPPB_ENABLE_OpenACC=ON \
  -DPPB_ENABLE_OpenMP=ON \
  -DCMAKE_C_COMPILER=nvc \
  -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_CUDA_HOST_COMPILER=nvc++ \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE86=ON

echo "Building all targets..."
cmake --build .