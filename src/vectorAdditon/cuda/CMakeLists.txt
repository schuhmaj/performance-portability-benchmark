find_package(LIKWID REQUIRED)

add_executable(vec_cuda main.cpp Impl_Cuda.cu)
target_link_libraries(vec_cuda PRIVATE vec_lib)
set_target_properties(vec_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(vec_cuda
    PRIVATE
        vec_lib
        CUDA::cublas
)
target_link_libraries(vec_cuda
        PRIVATE
        likwid::likwid
)