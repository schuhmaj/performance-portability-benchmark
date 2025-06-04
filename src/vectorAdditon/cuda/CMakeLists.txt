add_executable(vec_cuda Impl_Cuda.cu)
target_link_libraries(vec_cuda PRIVATE vec_lib)
set_target_properties(vec_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(vec_cublas Impl_Cublas.cu)
target_link_libraries(vec_cublas
    PRIVATE
        vec_lib
        CUDA::cublas
)
set_target_properties(vec_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)