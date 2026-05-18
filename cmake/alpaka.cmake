message(STATUS "Setting up Alpaka")
set(Alpaka_VERSION 2.0.0)

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

# Enable CUDA backend if CUDA language is enabled
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if ("CUDA" IN_LIST languages)
    set(alpaka_ACC_GPU_CUDA_ENABLE ON CACHE BOOL "Enable Alpaka CUDA backend" FORCE)
endif ()
