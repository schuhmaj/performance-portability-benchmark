message(STATUS "Setting up Alpaka")
set(Alpaka_VERSION 2.0.0)

# Enable CUDA backend if CUDA language is enabled
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if ("CUDA" IN_LIST languages)
    set(alpaka_ACC_GPU_CUDA_ENABLE ON CACHE BOOL "Enable Alpaka CUDA backend" FORCE)
endif ()
# Enable HIP backend if HIP language is enabled
if ("HIP" IN_LIST languages)
    set(alpaka_ACC_GPU_HIP_ENABLE ON CACHE BOOL "Enable Alpaka HIP backend" FORCE)
endif ()

# Enable SYCL backend when compiling with Intel's oneAPI compiler (IntelLLVM / icpx).
# SYCL is not a CMake "language", so we detect it via the C++ compiler id instead -
# this mirrors how alpaka itself gates its SYCL back-end.
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(alpaka_ACC_SYCL_ENABLE ON CACHE BOOL "Enable Alpaka SYCL backend" FORCE)
    set(alpaka_SYCL_ONEAPI_GPU ON CACHE BOOL "Enable oneAPI GPU target for the SYCL back-end" FORCE)
    set(alpaka_SYCL_ONEAPI_GPU_DEVICES "spir64" CACHE STRING "oneAPI GPU device target(s)")
endif ()

find_package(Alpaka ${Alpaka_VERSION} CONFIG QUIET)

if (${Alpaka_FOUND})
    message(STATUS "Found existing Alpaka libraries: ${Alpaka_DIR}")
else ()
    message(STATUS "Using Alpaka from GitHub Release ${Alpaka_VERSION}")
    include(FetchContent)

    FetchContent_Declare(
            Alpaka
            URL https://github.com/alpaka-group/alpaka/archive/refs/tags/${Alpaka_VERSION}.tar.gz
    )
    FetchContent_MakeAvailable(Alpaka)
endif ()
