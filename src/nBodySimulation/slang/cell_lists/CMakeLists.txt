set(SLANG_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelForce.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelVelocity.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelPosition.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelBlellochScan.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelBlockSum.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelIdCells.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelResetCells.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelHistogram.slang
)

if (${PPB_ENABLE_Slang_Vulkan})
    compile_slang(
            LANGUAGE SPIR-V
            INFILE ${SLANG_SOURCE_FILES}
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            OUT_HEADERS SLANG_VULKAN_HEADERS
    )

    set(CELLS_DIR ${NBODY_SIMULATION_ROOT_DIR}/vulkan/cell_lists)
    add_library(nbody_slang_vulkan_cells_lib STATIC
            ${CELLS_DIR}/Impl_Vulkan.h ${CELLS_DIR}/Impl_Vulkan.cpp ${CELLS_DIR}/Push_Constants.h
            ${SLANG_VULKAN_HEADERS}
    )

    target_compile_features(nbody_slang_vulkan_cells_lib PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_vulkan_cells_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
    target_include_directories(nbody_slang_vulkan_cells_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

    add_executable(nbody_slang_vulkan_cells ${CELLS_DIR}/main.cpp)
    target_compile_definitions(nbody_slang_vulkan_cells PRIVATE PARADIGM="Slang-Vulkan")
    target_link_libraries(nbody_slang_vulkan_cells PRIVATE nbody_slang_vulkan_cells_lib)

endif ()

if (${PPB_ENABLE_Slang_Cuda})
    compile_slang(
            LANGUAGE PTX
            INFILE ${SLANG_SOURCE_FILES}
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            OUT_HEADERS SLANG_PTX_FILES
    )

    add_library(nbody_slang_cuda_cells_lib STATIC
        Impl_Slang_Cuda.cuh Impl_Slang_Cuda.cu ${SLANG_PTX_FILES}
    )

    target_compile_definitions(nbody_slang_cuda_cells_lib PRIVATE
        SLANG_PTX_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    target_compile_features(nbody_slang_cuda_cells_lib PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_cuda_cells_lib PUBLIC nbody_lib CUDA::cudart CUDA::cuda_driver)
    target_compile_options(nbody_slang_cuda_cells_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    set_target_properties(nbody_slang_cuda_cells_lib PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )

    add_executable(nbody_slang_cuda_cells main.cu)
    target_compile_features(nbody_slang_cuda_cells PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_cuda_cells PRIVATE nbody_slang_cuda_cells_lib)
    set_target_properties(nbody_slang_cuda_cells PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )

endif()
