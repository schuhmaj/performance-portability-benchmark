add_executable(nbody_raja_test RajaNBodyTest.cpp RajaTestMain.cpp)
target_link_libraries(nbody_raja_test
    PRIVATE
        nbody_test_lib
        nbody_raja_lib
)
gtest_discover_tests(nbody_raja_test)
