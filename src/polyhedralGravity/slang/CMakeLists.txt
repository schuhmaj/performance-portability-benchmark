if (${PPB_Polyhedral_FloatType} EQUAL 32)
    set(SLANG_COMPILER_DEFINITIONS -DFloatType=float -DFloatType3=float3 -DFloatType4=float4 -DFloatTypeM=mat3)
elseif (${PPB_Polyhedral_FloatType} EQUAL 64)
    set(SLANG_COMPILER_DEFINITIONS -DFloatType=double -DFloatType3=double3 -DFloatType4=double4 -DFloatTypeM=dmat3)
endif ()

if (${PPB_ENABLE_Slang_Vulkan})
    compile_slang(
            LANGUAGE SPIR-V
            INFILE ${CMAKE_CURRENT_SOURCE_DIR}/shader/eval.slang
            KERNEL_NAME polyhedral_eval_kernel
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            OUT_HEADERS SLANG_VULKAN_HEADERS
            DEFINITIONS ${SLANG_COMPILER_DEFINITIONS}
    )
    add_library(polyhedral_slang_vulkan_lib STATIC Impl_Slang_Vulkan.cpp ${SLANG_VULKAN_HEADERS})
    target_compile_features(polyhedral_slang_vulkan_lib PUBLIC cxx_std_17)
    target_link_libraries(polyhedral_slang_vulkan_lib PUBLIC polyhedral_lib kompute::kompute glm::glm common_vulkan_lib)
    target_include_directories(polyhedral_slang_vulkan_lib PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

    add_executable(polyhedral_slang_vulkan ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
    target_link_libraries(polyhedral_slang_vulkan PRIVATE polyhedral_slang_vulkan_lib)
endif ()

if (${PPB_ENABLE_Slang_Cuda})
    compile_slang(
            LANGUAGE CUDA
            INFILE ${CMAKE_CURRENT_SOURCE_DIR}/shader/eval.slang
            PATH ${CMAKE_CURRENT_BINARY_DIR}
            KERNEL_NAME run_eval
            OUT_HEADERS SLANG_CUDA_HEADERS
            DEFINITIONS ${SLANG_COMPILER_DEFINITIONS}
    )
    add_library(polyhedral_slang_cuda_lib STATIC Impl_Slang_Cuda.cu wrapper_eval.cu ${SLANG_CUDA_HEADERS})
    target_include_directories(polyhedral_slang_cuda_lib PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(polyhedral_slang_cuda_lib PUBLIC polyhedral_lib)

    add_executable(polyhedral_slang_cuda ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
    target_link_libraries(polyhedral_slang_cuda PRIVATE polyhedral_slang_cuda_lib)
endif ()
