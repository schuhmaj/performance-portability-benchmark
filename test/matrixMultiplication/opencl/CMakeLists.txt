add_executable(matMul_ocl_test OpenCLMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_ocl_test
    PRIVATE
        matMul_test_lib
        matMul_ocl_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_ocl_test)