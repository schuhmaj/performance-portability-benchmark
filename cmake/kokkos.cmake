message(STATUS "Setting up Kokkos")
set(Kokkos_VERSION 5.1.1)

# Enable CUDA backend if CUDA language is enabled
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if ("CUDA" IN_LIST languages)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
elseif ("HIP" IN_LIST languages)
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable Kokkos HIP backend" FORCE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(Kokkos_ENABLE_SYCL ON CACHE BOOL "Enable Kokkos SYCL backend" FORCE)
elseif (APPLE)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
endif ()

find_package(Kokkos 4.6.02...${Kokkos_VERSION} QUIET)

if (${Kokkos_FOUND})
    message(STATUS "Found existing Kokkos libraries: ${Kokkos_DIR}")
else ()
    message(STATUS "Using Kokkos from GitHub Release ${Kokkos_VERSION}")
    include(FetchContent)

    # For the CPU Code always optimize for the machine being build on (use vectorization, etc.)
    set(Kokkos_ARCH_NATIVE ON CACHE STRING "Always build for the machine on which is being compiled" FORCE)

    FetchContent_Declare(
            Kokkos
            URL https://github.com/kokkos/kokkos/archive/refs/tags/${Kokkos_VERSION}.tar.gz
    )
    FetchContent_MakeAvailable(Kokkos)
endif ()
