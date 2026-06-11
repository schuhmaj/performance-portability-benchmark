add_executable(nbody_cuda_test_verlet_lists CudaNBodyTest.cu)
target_link_libraries(nbody_cuda_test_verlet_lists
    PRIVATE
        nbody_test_lib
        nbody_cuda_lib_verlet_lists
        GTest::gtest_main
)
gtest_discover_tests(nbody_cuda_test_verlet_lists)