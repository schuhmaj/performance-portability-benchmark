set(KERNELS
    KernelPosition
    KernelForce
    KernelVelocity
    KernelBlellochScan
    KernelBlockSum
    KernelCountNeighbors
    KernelVerlet
)

if (${PPB_ENABLE_Slang_Vulkan})

    # compile slang -> vulkan
    slang_vulkan_compile_shaders(${CMAKE_CURRENT_SOURCE_DIR}/shaders TARGETS HEADERS verlet ppb)
    
    set(VERLET_DIR ${NBODY_SIMULATION_ROOT_DIR}/vulkan/verlet_lists)
    add_library(nbody_slang_vulkan_verlet_lib STATIC
            ${VERLET_DIR}/Impl_Vulkan.h ${VERLET_DIR}/Impl_Vulkan.cpp ${VERLET_DIR}/Push_Constants.h
            ${HEADERS}
    )

    add_dependencies(nbody_slang_vulkan_verlet_lib ${TARGETS})

    target_compile_features(nbody_slang_vulkan_verlet_lib PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_vulkan_verlet_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
    target_include_directories(nbody_slang_vulkan_verlet_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

    add_executable(nbody_slang_vulkan_verlet ${VERLET_DIR}/main.cpp)
    target_link_libraries(nbody_slang_vulkan_verlet PRIVATE nbody_slang_vulkan_verlet_lib)

endif()

if (${PPB_ENABLE_Slang_Cuda})

    add_library(nbody_slang_cuda_verlet_lib STATIC
        Impl_Slang_Cuda.cuh Impl_Slang_Cuda.cu
    )

    # compile slang -> ptx
    slang_ptx_compile_shaders(${CMAKE_CURRENT_SOURCE_DIR}/shaders TARGETS verlet)
    add_dependencies(nbody_slang_cuda_verlet_lib ${TARGETS})

    target_compile_definitions(nbody_slang_cuda_verlet_lib PRIVATE
        SLANG_PTX_DIR="${CMAKE_CURRENT_BINARY_DIR}/generated_shaders"
    )

    target_compile_features(nbody_slang_cuda_verlet_lib PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_cuda_verlet_lib PUBLIC nbody_lib CUDA::cudart CUDA::cuda_driver)
    target_compile_options(nbody_slang_cuda_verlet_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    set_target_properties(nbody_slang_cuda_verlet_lib PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )


    add_executable(nbody_slang_cuda_verlet main.cu)
    target_compile_features(nbody_slang_cuda_verlet PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_cuda_verlet PRIVATE nbody_slang_cuda_verlet_lib)
    set_target_properties(nbody_slang_cuda_verlet PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )
    
endif()
