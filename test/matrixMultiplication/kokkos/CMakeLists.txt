add_executable(matMul_kokkos_test KokkosMatrixMultiplicationTest.cpp KokkosTestMain.cpp)
target_link_libraries(matMul_kokkos_test
    PRIVATE
        matMul_test_lib
        matMul_kokkos_lib
)
gtest_discover_tests(matMul_kokkos_test)