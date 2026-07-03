if (${PPB_ENABLE_DOMAIN_COLORING}) 
    add_compile_definitions(PPB_ENABLE_DOMAIN_COLORING)
endif ()


include_directories(include)
add_library(nbody_cuda_lib_linked_cells STATIC
        src/common.cu
        src/CudaParticleSoA.cu 
        src/Impl_Cuda.cu)
target_compile_features(nbody_cuda_lib_linked_cells PUBLIC cxx_std_20)
target_link_libraries(nbody_cuda_lib_linked_cells PUBLIC nbody_lib)
target_compile_options(nbody_cuda_lib_linked_cells PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
target_compile_options(nbody_cuda_lib_linked_cells PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(nbody_cuda_lib_linked_cells PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

add_executable(nbody_cuda_linked_cells src/main.cu)
target_compile_features(nbody_cuda_linked_cells PUBLIC cxx_std_20)
target_link_libraries(nbody_cuda_linked_cells PRIVATE nbody_cuda_lib_linked_cells)
set_target_properties(nbody_cuda_linked_cells PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)