alpaka_add_executable(matMul_alpaka_test AlpakaMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_alpaka_test
        PRIVATE
        matMul_test_lib
        matMul_alpaka_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_alpaka_test)