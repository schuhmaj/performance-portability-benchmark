add_executable(vec_cuda main.cpp Impl_Cuda.cu Impl_Cublas.cu Impl_ChunkedCuda.cu Impl_Thrust.cu Implementations.cuh)
target_link_libraries(vec_cuda PRIVATE vec_lib)
set_target_properties(vec_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(vec_cuda
    PRIVATE
        vec_lib
        CUDA::cublas
)