add_executable(matMul_stdpar_test StdparMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_stdpar_test
    PRIVATE
        matMul_test_lib
        matMul_stdpar_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_stdpar_test)
