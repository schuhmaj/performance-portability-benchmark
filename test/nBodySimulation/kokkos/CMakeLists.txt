add_executable(nbody_kokkos_test KokkosTest.cpp KokkosTestMain.cpp)
target_link_libraries(nbody_kokkos_test
    PRIVATE
        nbody_test_lib
        nbody_kokkos_lib
)
gtest_discover_tests(nbody_kokkos_test)