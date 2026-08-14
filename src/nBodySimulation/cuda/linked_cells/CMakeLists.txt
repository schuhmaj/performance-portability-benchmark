add_library(nbody_cuda_lib_linked_cells STATIC
        src/kernels.cu
        src/CudaParticleSoA.cu 
        src/Impl_Cuda.cu)
target_include_directories(nbody_cuda_lib_linked_cells PUBLIC include)
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

if (${PPB_ENABLE_CUDA_DOMAIN_COLORING}) 
    target_compile_definitions(nbody_cuda_lib_linked_cells PUBLIC PPB_ENABLE_CUDA_DOMAIN_COLORING)
endif ()
if (${PPB_ENABLE_CUDA_LINKED_CELL_OPTIMIZATION}) 
    target_compile_definitions(nbody_cuda_lib_linked_cells PUBLIC PPB_ENABLE_CUDA_LINKED_CELL_OPTIMIZATION)
endif ()
if (${PPB_ENABLE_CUDA_LINKED_CELL_OPTIMIZATION_ALT}) 
    target_compile_definitions(nbody_cuda_lib_linked_cells PUBLIC PPB_ENABLE_CUDA_LINKED_CELL_OPTIMIZATION_ALT)
endif ()
if (${PPB_ENABLE_CUDA_LINKED_CELL_BEST_OPTIMIZATION}) 
    target_compile_definitions(nbody_cuda_lib_linked_cells PUBLIC PPB_ENABLE_CUDA_LINKED_CELL_BEST_OPTIMIZATION)
endif ()
