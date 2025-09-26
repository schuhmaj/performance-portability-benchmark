compile_opencl(
        INFILE VectorAddition.cl
        OUTFILE VectorAdditionKernel.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
)
add_executable(vec_ocl Impl_OpenCL.cpp)
target_compile_features(vec_ocl PUBLIC cxx_std_17)
target_link_libraries(vec_ocl
    PRIVATE
        vec_lib
        common_ocl_lib
)
target_include_directories(vec_ocl PRIVATE ${CMAKE_CURRENT_BINARY_DIR})