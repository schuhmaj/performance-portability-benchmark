add_executable(nbody_hip_test HipNBodyTest.hip)
target_link_libraries(nbody_hip_test
    PRIVATE
        nbody_test_lib
        nbody_hip_lib
        GTest::gtest_main
)
gtest_discover_tests(nbody_hip_test)
