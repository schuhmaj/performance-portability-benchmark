add_executable(matMul_slang_vulkan_test SlangVulkanMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_slang_vulkan_test
    PRIVATE
        matMul_test_lib
        matMul_slang_vulkan_lib
        GTest::gtest_main
)
gtest_discover_tests(matMul_slang_vulkan_test)
