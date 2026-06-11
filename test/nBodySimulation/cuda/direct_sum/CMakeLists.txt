add_executable(nbody_cuda_test_direct_sum CudaNBodyTest.cu)
target_link_libraries(nbody_cuda_test_direct_sum
    PRIVATE
        nbody_test_lib
        nbody_cuda_lib_direct_sum
        GTest::gtest_main
)
gtest_discover_tests(nbody_cuda_test_direct_sum)