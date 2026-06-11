add_executable(nbody_cuda_test_linked_cells CudaNBodyTest.cu)
target_link_libraries(nbody_cuda_test_linked_cells
    PRIVATE
        nbody_test_lib
        nbody_cuda_lib_linked_cells
        GTest::gtest_main
)
gtest_discover_tests(nbody_cuda_test_linked_cells)