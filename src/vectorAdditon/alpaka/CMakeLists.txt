alpaka_add_executable(vec_alpaka Impl_Alpaka.cpp)
target_compile_features(vec_alpaka PUBLIC cxx_std_17)
target_link_libraries(vec_alpaka PRIVATE vec_lib alpaka::alpaka)