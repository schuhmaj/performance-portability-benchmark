add_executable(vec_ocl Impl_OpenCL.cpp)
target_compile_features(vec_ocl PUBLIC cxx_std_17)
target_link_libraries(vec_ocl
    PRIVATE
        vec_lib
        common_ocl_lib
)