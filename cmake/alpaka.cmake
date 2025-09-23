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