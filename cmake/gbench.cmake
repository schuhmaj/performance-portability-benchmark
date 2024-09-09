message(STATUS "Setting up Google Benchmark")

# Specify the Google Benchmark version you want to use
set(GOOGLEBENCH_VERSION 1.9.0)

# Try to find an existing installation of Google Benchmark
find_package(benchmark ${GOOGLEBENCH_VERSION})

if (benchmark_FOUND)
        message(STATUS "Found existing Google Benchmark libraries: ${benchmark_DIR}")
else ()
        message(STATUS "Using Google Benchmark from GitHub release ${GOOGLEBENCH_VERSION}")

        # Declare Google Benchmark to be fetched from GitHub
        FetchContent_Declare(googlebench
                GIT_REPOSITORY https://github.com/google/benchmark.git
                GIT_TAG v1.9.0
        )

        # Optionally disable testing for Google Benchmark
        option(BENCHMARK_ENABLE_TESTING "" OFF)

        # Fetch the content and make it available
        FetchContent_MakeAvailable(googlebenchmark)
endif ()