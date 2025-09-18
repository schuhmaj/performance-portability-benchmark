add_executable(matMul_omp_test OpenMPMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_omp_test
    PRIVATE
        matMul_test_lib
        matMul_omp_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_omp_test)