add_executable(matMul_acc_test OpenACCMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_acc_test
    PRIVATE
        matMul_test_lib
        matMul_acc_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_acc_test)