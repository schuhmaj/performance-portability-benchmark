add_executable(nbody_cuda_test CudaNBodyTest.cu)
target_link_libraries(nbody_cuda_test
    PRIVATE
        nbody_test_lib
        nbody_cuda_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_cuda_test)