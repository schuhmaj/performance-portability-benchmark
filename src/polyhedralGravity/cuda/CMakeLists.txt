add_library(polyhedral_cuda_lib STATIC Impl_Cuda.cu)
target_link_libraries(polyhedral_cuda_lib PUBLIC polyhedral_lib)
target_compile_features(polyhedral_cuda_lib PUBLIC cxx_std_17)
target_compile_options(polyhedral_cuda_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
set_target_properties(polyhedral_cuda_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)


add_executable(polyhedral_cuda ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_link_libraries(polyhedral_cuda PRIVATE polyhedral_cuda_lib)
target_compile_features(polyhedral_cuda PUBLIC cxx_std_17)
target_compile_options(polyhedral_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
set_target_properties(polyhedral_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)


