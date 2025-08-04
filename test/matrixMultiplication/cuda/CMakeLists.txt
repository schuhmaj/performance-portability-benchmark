add_executable(matMul_cuda_test CudaMatrixMultiplicationTest.cu)
target_link_libraries(matMul_cuda_test
    PRIVATE
        matMul_test_lib
        matMul_cuda_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_cuda_test)