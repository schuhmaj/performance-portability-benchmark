message(STATUS "Setting up Vulkan Kompute")
set(Kompute_VERSION 0.9.0)

find_package(Vulkan REQUIRED)
find_package(kompute ${Kompute_VERSION} QUIET)

if (${kompute_FOUND})
    message(STATUS "Found existing Vulkan Kompute libraries: ${kompute_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${kompute_DIR}")
else ()
    message(STATUS "Using Vulkan Kompute from GitHub Release ${Kompute_VERSION}")
    include(FetchContent)

    FetchContent_Declare(
            Kompute
            URL https://github.com/KomputeProject/kompute/archive/refs/tags/v${Kompute_VERSION}.tar.gz
    )

    set(KOMPUTE_OPT_LOG_LEVEL "Off" CACHE STRING "Kompute log level" FORCE)

    FetchContent_MakeAvailable(Kompute)
    target_compile_options(kompute PRIVATE
            -Wno-error
            -Wno-deprecated-literal-operator
            -Wno-deprecated
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-sign-compare
    )
endif ()

list(APPEND CMAKE_PREFIX_PATH "${kompute_SOURCE_DIR}/cmake")