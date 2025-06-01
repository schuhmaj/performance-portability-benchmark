add_executable(vec_acpp Impl_AdaptiveCpp.cpp)
target_compile_features(vec_acpp PUBLIC cxx_std_17)
add_sycl_to_target(TARGET vec_acpp)
target_link_libraries(vec_acpp PRIVATE vec_lib)

#add_executable(vec_acpp_std Impl_AdaptiveCppStd.cpp)
#target_compile_features(vec_acpp_std PUBLIC cxx_std_17)
#target_include_directories(vec_acpp_std PUBLIC ${VECTOR_ADDITION_ROOT_DIR})
#target_compile_options(vec_acpp_std PRIVATE --acpp-stdpar --acpp-stdpar-unconditional-offload)
#add_sycl_to_target(TARGET vec_acpp_std)
#target_link_libraries(vec_acpp_std
#        PRIVATE
#        benchmark::benchmark
#        TBB::tbb
#)