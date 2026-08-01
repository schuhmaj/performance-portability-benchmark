add_executable(nbody_vulkan_naive_test VulkanNBodyTest.cpp)
target_link_libraries(nbody_vulkan_naive_test
    PRIVATE
        nbody_test_lib
        nbody_vulkan_naive_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_vulkan_naive_test)