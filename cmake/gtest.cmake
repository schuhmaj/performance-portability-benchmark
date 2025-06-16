include(FetchContent)

message(STATUS "Setting up gtest")

include(FetchContent)

message(STATUS "Setting up Google Benchmark")
set(GOOGLE_TEST_VERSION 1.17.0)

find_package(GTest ${GOOGLE_BENCHMARK_VERSION} QUIET)

if (${GTest_FOUND})
        message(STATUS "Found existing Google Test: ${GTEST_INCLUDE_DIRS}")
else ()
        message(STATUS "Using Google Test from GitHub Release ${GOOGLE_TEST_VERSION}")
        FetchContent_Declare(googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v${GOOGLE_TEST_VERSION}
        )
        FetchContent_MakeAvailable(googletest)
        target_compile_options(gtest_main PRIVATE -w)
        get_target_property(propval gtest_main INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(gtest_main SYSTEM PUBLIC "${propval}")
endif ()
