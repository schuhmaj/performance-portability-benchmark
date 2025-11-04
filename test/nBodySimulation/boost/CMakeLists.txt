add_executable(nbody_boost_test BoostNBodyTest.cpp)
target_link_libraries(nbody_boost_test
    PRIVATE
        nbody_test_lib
        nbody_boost_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_boost_test)