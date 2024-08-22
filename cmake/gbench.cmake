include(FetchContent)

message(STATUS "Setting up googlebench")

FetchContent_Declare(googlebench
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.4
        )

option(BENCHMARK_ENABLE_TESTING "" OFF)

FetchContent_MakeAvailable(googlebench)
