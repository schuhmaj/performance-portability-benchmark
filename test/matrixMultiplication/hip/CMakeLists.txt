add_executable(matMul_hip_test HipMatrixMultiplicationTest.hip)
target_link_libraries(matMul_hip_test
    PRIVATE
        matMul_test_lib
        matMul_hip_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_hip_test)
