if (${PPB_ENABLE_Slang_Vulkan})
    compile_slang(
            LANGUAGE SPIR-V
            INFILE ${CMAKE_CURRENT_SOURCE_DIR}/VectorAdditionShader.slang
            KERNEL_NAME vecadd_kernel
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            OUT_HEADERS SLANG_VULKAN_HEADERS
            DEFINITIONS ${SLANG_COMPILER_DEFINITIONS}
    )

    add_executable(vec_slang_vulkan Impl_SlangVulkan.cpp ${SLANG_VULKAN_HEADERS})
    target_compile_features(vec_slang_vulkan PUBLIC cxx_std_17)
    target_link_libraries(vec_slang_vulkan PUBLIC vec_lib kompute::kompute common_vulkan_lib)
    target_include_directories(vec_slang_vulkan PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endif ()

if (${PPB_ENABLE_Slang_Cuda})
    compile_slang(
            LANGUAGE PTX
            INFILE ${CMAKE_CURRENT_SOURCE_DIR}/VectorAdditionShader.slang
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            KERNEL_NAME vecadd_kernel
            OUT_HEADERS SLANG_PTX_FILES
            DEFINITIONS ${SLANG_COMPILER_DEFINITIONS}
    )
    add_executable(vec_slang_cuda
            Impl_SlangCuda.cu ${SLANG_PTX_FILES}
    )

    target_compile_definitions(vec_slang_cuda PRIVATE
            SLANG_PTX_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    target_compile_features(vec_slang_cuda PUBLIC cxx_std_20)
    target_link_libraries(vec_slang_cuda PUBLIC vec_lib CUDA::cudart CUDA::cuda_driver)
    target_compile_options(vec_slang_cuda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    set_target_properties(vec_slang_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )
endif ()
