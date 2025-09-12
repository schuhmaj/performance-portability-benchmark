add_executable(vec_cpp Impl_Cpp.cpp)
target_compile_features(vec_cpp PUBLIC cxx_std_17)
target_link_libraries(vec_cpp PUBLIC vec_lib)