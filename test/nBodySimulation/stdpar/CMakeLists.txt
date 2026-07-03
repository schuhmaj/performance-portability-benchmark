add_executable(nbody_stdpar_test StdparNBodyTest.cpp)
target_link_libraries(nbody_stdpar_test
    PRIVATE
        nbody_test_lib
        nbody_stdpar_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_stdpar_test)
