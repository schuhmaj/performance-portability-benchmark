message(STATUS "Setting up Vulkan Kompute")
set(Kompute_VERSION 0.9.0)

find_package(Vulkan REQUIRED)
find_package(Kompute ${Kompute_VERSION} CONFIG QUIET)

if (${Kompute_FOUND})
    message(STATUS "Found existing Vulkan Kompute libraries: ${Kompute_DIR}")
else ()
    message(STATUS "Using Vulkan Kompute from GitHub Release ${Kompute_VERSION}")
    include(FetchContent)

    FetchContent_Declare(
            Kompute
            URL https://github.com/KomputeProject/kompute/archive/refs/tags/v${Kompute_VERSION}.tar.gz
    )
    FetchContent_MakeAvailable(Kompute)
endif ()

list(APPEND CMAKE_PREFIX_PATH "${kompute_SOURCE_DIR}/cmake")