add_library(matMul_cuda_lib STATIC Impl_Cuda.cu Impl_Cuda.cuh Impl_CudaTensor.cu Impl_CudaTensor.cuh Common.cuh Impl_CudaBuffer.cu Impl_CudaBuffer.cuh)
target_compile_features(matMul_cuda_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_cuda_lib PUBLIC matMul_lib)
target_compile_options(matMul_cuda_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
set_target_properties(matMul_cuda_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)


add_executable(matMul_cuda main.cu)
target_link_libraries(matMul_cuda PRIVATE matMul_cuda_lib)
target_compile_features(matMul_cuda PUBLIC cxx_std_17)
target_compile_options(matMul_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
set_target_properties(matMul_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

