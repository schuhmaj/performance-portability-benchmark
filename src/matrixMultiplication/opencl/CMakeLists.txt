compile_opencl(
    INFILE MatrixMultiplication.cl
    OUTFILE MatrixMultiplicationKernel.h
    PATH ${CMAKE_CURRENT_BINARY_DIR}
)
add_library(matMul_ocl_lib STATIC Impl_OpenCL.cpp Impl_OpenCL.h "${CMAKE_CURRENT_BINARY_DIR}/MatrixMultiplicationKernel.h")
target_compile_features(matMul_ocl_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_ocl_lib
    PUBLIC
        matMul_lib
        common_ocl_lib
)
target_include_directories(matMul_ocl_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})


add_executable(matMul_ocl main.cpp)
target_link_libraries(matMul_ocl matMul_ocl_lib)