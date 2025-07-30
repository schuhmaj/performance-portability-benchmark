add_executable(matMul_cublas Impl_Cublas.cuh Impl_Cublas.cu main.cu)
target_link_libraries(matMul_cublas
        PRIVATE
        matMul_lib
        CUDA::cublas
)
target_compile_features(matMul_cublas PUBLIC cxx_std_17)
set_target_properties(matMul_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)