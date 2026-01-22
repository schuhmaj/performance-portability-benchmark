add_executable(matMul_vulkan_test VulkanMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_vulkan_test
    PRIVATE
        matMul_test_lib
        matMul_vulkan_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_vulkan_test)