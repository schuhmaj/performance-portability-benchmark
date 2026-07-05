add_executable(matMul_raja_test RajaMatrixMultiplicationTest.cpp RajaTestMain.cpp)
target_link_libraries(matMul_raja_test
    PRIVATE
        matMul_test_lib
        matMul_raja_lib
)
gtest_discover_tests(matMul_raja_test)
