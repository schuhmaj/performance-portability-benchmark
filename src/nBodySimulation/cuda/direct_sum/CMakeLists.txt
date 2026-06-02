add_library(nbody_cuda_lib_direct_sum STATIC Impl_Cuda.cu Impl_Cuda.cuh)
target_compile_features(nbody_cuda_lib_direct_sum PUBLIC cxx_std_20)
target_link_libraries(nbody_cuda_lib_direct_sum PUBLIC nbody_lib)
target_compile_options(nbody_cuda_lib_direct_sum PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
set_target_properties(nbody_cuda_lib_direct_sum PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

add_executable(nbody_cuda_direct_sum main.cu)
target_compile_features(nbody_cuda_direct_sum PUBLIC cxx_std_20)
target_link_libraries(nbody_cuda_direct_sum PRIVATE nbody_cuda_lib_direct_sum)
set_target_properties(nbody_cuda_direct_sum PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)