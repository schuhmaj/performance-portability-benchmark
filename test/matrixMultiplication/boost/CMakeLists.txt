add_executable(matMul_boost_test BoostMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_boost_test
    PRIVATE
        matMul_test_lib
        matMul_boost_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_boost_test)