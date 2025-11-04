add_executable(matMul_acpp_test AdaptiveCppMatrixMultiplicationTest.cpp)
target_link_libraries(matMul_acpp_test
    PRIVATE
        matMul_test_lib
        matMul_acpp_lib
        GTest::gtest_main
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(matMul_acpp_test PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    target_link_options(matMul_acpp_test PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
else ()
    add_sycl_to_target(TARGET matMul_acpp_test)
endif ()

gtest_discover_tests(matMul_acpp_test)