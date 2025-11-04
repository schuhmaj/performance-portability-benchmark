add_executable(nbody_cpp_test CppNBodyTest.cpp)
target_link_libraries(nbody_cpp_test
    PRIVATE
        nbody_test_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_cpp_test)