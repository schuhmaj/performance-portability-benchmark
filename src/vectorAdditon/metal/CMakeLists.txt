add_executable(vec_metal Impl_Metal.cpp)
target_compile_features(vec_metal PUBLIC cxx_std_17)
target_link_libraries(vec_metal
    PRIVATE
        vec_lib
        MetalCpp::metal-cpp
)