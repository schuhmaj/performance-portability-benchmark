add_executable(matMul_cuda main.cu Impl_Cuda.cu Impl_Cuda.cuh)
target_link_libraries(matMul_cuda PRIVATE matMul_lib)
target_compile_features(matMul_cuda PUBLIC cxx_std_17)
set_target_properties(matMul_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
