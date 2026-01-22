add_executable(nbody_omp_test OpenMPNBodyTest.cpp)
target_link_libraries(nbody_omp_test
    PRIVATE
        nbody_test_lib
        nbody_omp_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_omp_test)