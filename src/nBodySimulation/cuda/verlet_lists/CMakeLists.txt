if (${PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS}) 
    add_compile_definitions(PPB_ENABLE_VERLET_CLUSTER_LISTS)
endif ()

if (${PPB_ENABLE_CUDA_VERLET_LISTS_LC_OPTIMIZATION}) 
    add_compile_definitions(PPB_ENABLE_VERLET_LISTS_LC_OPTIMIZATION)
endif ()

include_directories(include)
add_library(nbody_cuda_lib_verlet_lists STATIC
        src/common.cu
        src/CudaParticleSoA.cu 
        src/Impl_Cuda.cu)
target_compile_features(nbody_cuda_lib_verlet_lists PUBLIC cxx_std_20)
target_compile_options(nbody_cuda_lib_verlet_lists PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
target_compile_options(nbody_cuda_lib_verlet_lists PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(nbody_cuda_lib_verlet_lists PUBLIC nbody_lib)
set_target_properties(nbody_cuda_lib_linked_cells PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

add_executable(nbody_cuda_verlet_lists src/main.cu)
target_compile_features(nbody_cuda_verlet_lists PUBLIC cxx_std_20)
target_link_libraries(nbody_cuda_verlet_lists PRIVATE nbody_cuda_lib_verlet_lists)
set_target_properties(nbody_cuda_verlet_lists PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)