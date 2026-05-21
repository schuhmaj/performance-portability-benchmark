if (${PPB_Polyhedral_FloatType} EQUAL 32)
    set(SLANG_COMPILER_DEFINITIONS -DFloatType=float -DFloatType3=float3 -DFloatType4=float4 -DFloatTypeM=mat3)
elseif (${PPB_Polyhedral_FloatType} EQUAL 64)
    set(SLANG_COMPILER_DEFINITIONS -DFloatType=double -DFloatType3=double3 -DFloatType4=double4 -DFloatTypeM=dmat3)
endif ()

if (${PPB_ENABLE_Slang_Vulkan})
    compile_slang(
            LANGUAGE SPIR-V
            INFILE ${CMAKE_CURRENT_SOURCE_DIR}/MatrixMultiplicationShader.slang
            KERNEL_NAME matmul_kernel
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            OUT_HEADERS SLANG_VULKAN_HEADERS
            DEFINITIONS ${SLANG_COMPILER_DEFINITIONS}
    )

    add_library(matMul_slang_vulkan_lib STATIC Impl_SlangVulkan.h Impl_SlangVulkan.cpp ${SLANG_VULKAN_HEADERS})
    target_compile_features(matMul_slang_vulkan_lib PUBLIC cxx_std_17)
    target_link_libraries(matMul_slang_vulkan_lib PUBLIC matMul_lib kompute::kompute common_vulkan_lib)
    target_include_directories(matMul_slang_vulkan_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

    add_executable(matMul_slang_vulkan main.cpp)
    target_link_libraries(matMul_slang_vulkan PRIVATE matMul_slang_vulkan_lib)
endif ()

if (${PPB_ENABLE_Slang_Cuda})
    compile_slang(
            LANGUAGE PTX
            INFILE ${CMAKE_CURRENT_SOURCE_DIR}/MatrixMultiplicationShader.slang
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            KERNEL_NAME matmul_kernel
            OUT_HEADERS SLANG_PTX_FILES
            DEFINITIONS ${SLANG_COMPILER_DEFINITIONS}
    )
    add_library(matMul_slang_cuda_lib STATIC
            Impl_SlangCuda.cuh Impl_SlangCuda.cu ${SLANG_PTX_FILES}
    )

    target_compile_definitions(matMul_slang_cuda_lib PRIVATE
            SLANG_PTX_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

    target_compile_features(matMul_slang_cuda_lib PUBLIC cxx_std_20)
    target_link_libraries(matMul_slang_cuda_lib PUBLIC matMul_lib CUDA::cudart CUDA::cuda_driver)
    target_compile_options(matMul_slang_cuda_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    set_target_properties(matMul_slang_cuda_lib PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )

    add_executable(matMul_slang_cuda main.cu)
    target_compile_features(matMul_slang_cuda PUBLIC cxx_std_20)
    target_link_libraries(matMul_slang_cuda PRIVATE matMul_slang_cuda_lib)
    set_target_properties(matMul_slang_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )
endif ()
