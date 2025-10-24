add_executable(nbody_vulkan_test VulkanNBodyTest.cpp)
target_link_libraries(nbody_vulkan_test
    PRIVATE
        nbody_test_lib
        nbody_vulkan_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_vulkan_test)