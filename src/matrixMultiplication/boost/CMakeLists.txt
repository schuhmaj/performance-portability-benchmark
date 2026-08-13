set(MATMUL_OCL_KERNEL_DIR ${MATRIX_MULTIPLICATION_ROOT_DIR}/opencl/)

compile_opencl(
        INFILE ${MATMUL_OCL_KERNEL_DIR}/MatrixMultiplication.cl
        OUTFILE MatrixMultiplicationKernel.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
)
add_library(matMul_boost_lib STATIC Impl_Boost.cpp Impl_Boost.h "${CMAKE_CURRENT_BINARY_DIR}/MatrixMultiplicationKernel.h")
target_compile_features(matMul_boost_lib PUBLIC cxx_std_17)
target_include_directories(matMul_boost_lib PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(matMul_boost_lib
        PUBLIC
        OpenCL::OpenCL
        matMul_lib
)
target_include_directories(matMul_boost_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_executable(matMul_boost main.cpp)
target_link_libraries(matMul_boost PUBLIC matMul_boost_lib)