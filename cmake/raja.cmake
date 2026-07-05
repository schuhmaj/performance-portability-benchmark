message(STATUS "Setting up RAJA")
set(RAJA_VERSION 2025.12.2)

# Enable device backend if the corresponding language/ compiler is enabled (analogous to kokkos.cmake)
# The BLT (ENABLE_*) variants are set as well since RAJA's build system defaults its
# RAJA_ENABLE_* options to the BLT ones
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if ("CUDA" IN_LIST languages)
    set(RAJA_ENABLE_CUDA ON CACHE BOOL "Enable RAJA CUDA backend" FORCE)
    set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA (BLT)" FORCE)
    # BLT (RAJA's build system) refuses to configure when CMAKE_CUDA_HOST_COMPILER is unset.
    # The CUDA language is already enabled at this point, so this assignment does not alter
    # the compile rules anymore (nvcc keeps its default host compiler like for all other
    # CUDA targets of this project) - it merely satisfies the BLT sanity check.
    if (NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif ()
    # RAJA requires an external CUB for CUDA >= 11, but its FindCUB module does not know
    # about the cccl subdirectory the CUDA Toolkit >= 13 ships CUB in
    if (NOT CUB_DIR)
        foreach (cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
            if (EXISTS ${cuda_include_dir}/cub/cub.cuh)
                set(CUB_DIR ${cuda_include_dir} CACHE PATH "CUB include directory" FORCE)
            elseif (EXISTS ${cuda_include_dir}/cccl/cub/cub.cuh)
                set(CUB_DIR ${cuda_include_dir}/cccl CACHE PATH "CUB include directory" FORCE)
            endif ()
        endforeach ()
    endif ()
elseif ("HIP" IN_LIST languages)
    set(RAJA_ENABLE_HIP ON CACHE BOOL "Enable RAJA HIP backend" FORCE)
    set(ENABLE_HIP ON CACHE BOOL "Enable HIP (BLT)" FORCE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(RAJA_ENABLE_SYCL ON CACHE BOOL "Enable RAJA SYCL backend" FORCE)
elseif (APPLE)
    set(RAJA_ENABLE_OPENMP ON CACHE BOOL "Enable RAJA OpenMP backend" FORCE)
    set(ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP (BLT)" FORCE)
endif ()

find_package(RAJA ${RAJA_VERSION} CONFIG QUIET)

if (${RAJA_FOUND})
    message(STATUS "Found existing RAJA installation: ${RAJA_DIR}")
else ()
    message(STATUS "Using RAJA from GitHub Release ${RAJA_VERSION}")
    include(FetchContent)

    set(RAJA_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_EXERCISES OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(RAJA_ENABLE_DOCUMENTATION OFF CACHE BOOL "" FORCE)
    set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ENABLE_DOCS OFF CACHE BOOL "" FORCE)

    # The GitHub Release tarball already bundles the BLT and camp submodules
    # The patch step fixes the cudaMemAdvise call for the CUDA Toolkit >= 13 (no-op otherwise)
    FetchContent_Declare(
            RAJA
            URL https://github.com/LLNL/RAJA/releases/download/v${RAJA_VERSION}/RAJA-v${RAJA_VERSION}.tar.gz
            PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/scripts/patch_raja_cuda13.cmake
    )
    FetchContent_MakeAvailable(RAJA)

    # BLT (RAJA's build system) caches CMAKE_RUNTIME_OUTPUT_DIRECTORY/ LIBRARY_OUTPUT_PATH
    # pointing into RAJA's binary dir, which would relocate the outputs of every target of
    # this project configured afterwards - undo this (RAJA's own targets are unaffected)
    unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY CACHE)
    unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    unset(LIBRARY_OUTPUT_PATH CACHE)
    unset(LIBRARY_OUTPUT_PATH)
endif ()

# Unlike Kokkos, RAJA does not forward device compilation to consuming targets.
# Sources containing RAJA kernels must be compiled by the device compiler themselves
# when a GPU backend is active. Call this on all sources containing RAJA kernels.
function(ppb_raja_device_sources)
    if (RAJA_ENABLE_CUDA)
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    elseif (RAJA_ENABLE_HIP)
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
    endif ()
endfunction()

# Sets the compile/ link options required by the active RAJA device backend on a target
# (device lambdas for nvcc, SYCL offload for IntelLLVM)
function(ppb_raja_target_options target)
    if (RAJA_ENABLE_CUDA)
        target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    elseif (RAJA_ENABLE_SYCL)
        target_compile_options(${target} PRIVATE -fsycl)
        target_link_options(${target} PRIVATE -fsycl)
    endif ()
endfunction()
