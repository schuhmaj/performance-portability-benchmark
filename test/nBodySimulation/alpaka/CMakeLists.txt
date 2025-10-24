alpaka_add_executable(nbody_alpaka_test AlpakaNBodyTest.cpp)
target_link_libraries(nbody_alpaka_test
    PRIVATE
        nbody_test_lib
        nbody_alpaka_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_alpaka_test)