add_executable(vec_metal Impl_Metal.cpp)
target_include_directories(vec_metal PUBLIC ${VECTOR_ADDITION_ROOT_DIR})
target_compile_features(vec_metal PUBLIC cxx_std_17)
target_link_libraries(vec_metal
        PRIVATE
        benchmark::benchmark
        MetalCpp::metal-cpp
)