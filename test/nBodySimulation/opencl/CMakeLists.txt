add_executable(nbody_ocl_test OpenCLNBodyTest.cpp)
target_link_libraries(nbody_ocl_test
    PRIVATE
        nbody_test_lib
        nbody_ocl_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_ocl_test)