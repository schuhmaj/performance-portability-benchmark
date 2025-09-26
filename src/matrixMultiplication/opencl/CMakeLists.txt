compile_opencl(
    INFILE MatrixMultiplication.cl
    OUTFILE MatrixMultiplicationKernel.h
    PATH ${CMAKE_CURRENT_BINARY_DIR}
)
add_executable(matMul_ocl main.cpp Impl_OpenCL.cpp Impl_OpenCL.h)
target_compile_features(matMul_ocl PUBLIC cxx_std_17)
target_link_libraries(matMul_ocl
    PRIVATE
        matMul_lib
        common_ocl_lib
)
target_include_directories(matMul_ocl PRIVATE ${CMAKE_CURRENT_BINARY_DIR})