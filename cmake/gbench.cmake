include(FetchContent)

message(STATUS "Setting up Google Benchmark")
set(GOOGLE_BENCHMARK_VERSION 1.9.4)

find_package(benchmark ${GOOGLE_BENCHMARK_VERSION} QUIET)

if (${benchmark_FOUND})
    message(STATUS "Found existing Google Benchmark: ${benchmark_DIR}")
    set(GOOGLE_BENCHMARK_INCLUDE_DIR ${benchmark}/include)
    set(GOOGLE_BENCHMARK_LIBRARY_DIR ${benchmark}/lib)
else ()
    message(STATUS "Using Google Benchmark from GitHub Release ${GOOGLE_BENCHMARK_VERSION}")

    FetchContent_Declare(benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v${GOOGLE_BENCHMARK_VERSION}
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
    # Google Benchmark builds its own sources with -Werror -pedantic-errors. Newer
    # Clang-based compilers (e.g. IntelLLVM 2026 / Clang 26) flag __COUNTER__ as a
    # "C2y extension". -Werror is disabled here; -pedantic-errors (which Benchmark
    # adds unconditionally and exposes no option for) is neutralised for that one
    # diagnostic on the target below.
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Disable -Werror for Google Benchmark" FORCE)

    FetchContent_MakeAvailable(benchmark)

    # -pedantic-errors promotes the __COUNTER__ "C2y extension" warning to an error
    # independently of -Werror, so silence that specific group on Clang-based
    # toolchains. Appended last, so it overrides Benchmark's own -pedantic-errors.
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
        foreach (benchmark_target benchmark benchmark_main)
            if (TARGET ${benchmark_target})
                target_compile_options(${benchmark_target} PRIVATE -Wno-c2y-extensions)
            endif ()
        endforeach ()
    endif ()
    set(GOOGLE_BENCHMARK_INCLUDE_DIR ${benchmark_SOURCE_DIR}/include)
    set(GOOGLE_BENCHMARK_LIBRARY_DIR ${benchmark_BINARY_DIR}/src)
endif ()