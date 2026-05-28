add_executable(matMul_slang_cuda_test SlangCudaMatrixMultiplicationTest.cu)
target_compile_features(matMul_slang_cuda_test PUBLIC cxx_std_20)
target_link_libraries(matMul_slang_cuda_test
    PRIVATE
        matMul_test_lib
        matMul_slang_cuda_lib
        GTest::gtest_main
)
set_target_properties(matMul_slang_cuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)
gtest_discover_tests(matMul_slang_cuda_test)
