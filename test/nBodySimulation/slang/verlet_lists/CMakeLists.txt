add_executable(nbody_slang_vulkan_verlet_test SlangVulkanNBodyTest.cpp)
target_link_libraries(nbody_slang_vulkan_verlet_test
    PRIVATE
        nbody_test_lib
        nbody_slang_vulkan_verlet_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_slang_vulkan_verlet_test)