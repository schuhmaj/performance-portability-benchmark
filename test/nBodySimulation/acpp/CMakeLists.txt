add_executable(nbody_acpp_test AdaptiveCppNBodyTest.cpp)
target_link_libraries(nbody_acpp_test
        PRIVATE
        nbody_test_lib
        nbody_acpp_lib
        GTest::gtest_main
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_link_libraries(nbody_acpp_test PRIVATE ppb_sycl_offload)
else ()
    add_sycl_to_target(TARGET nbody_acpp_test)
endif ()

gtest_discover_tests(nbody_acpp_test)