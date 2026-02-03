add_executable(nbody_slang_cuda_cells_test SlangCudaNBodyTest.cu)
target_link_libraries(nbody_slang_cuda_cells_test
    PRIVATE
        nbody_test_lib
        nbody_slang_cuda_cells_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_slang_cuda_cells_test)