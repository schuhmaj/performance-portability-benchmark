add_executable(matMul_cpp_test CppMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_cpp_test
    PRIVATE
        matMul_test_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_cpp_test)