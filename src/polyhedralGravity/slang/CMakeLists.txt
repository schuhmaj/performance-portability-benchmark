function(slang_compile_shader infile)
    find_program(SLANG_COMPILER_PATH NAMES slangc)

    cmake_path(GET infile PARENT_PATH parent)
    cmake_path(GET infile STEM LAST_ONLY stem)

    cmake_path(ABSOLUTE_PATH infile OUTPUT_VARIABLE abs_path)

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shader")

    set(extra_arg "")
    if (FLOAT_BITS EQUAL 32)
        set(extra_arg -DFloatType=float -DFloatType3=float3 -DFloatType4=float4 -DFloatTypeM=mat3)
    endif()

    add_custom_command(
        OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.hpp"
        COMMAND
            ${SLANG_COMPILER_PATH}
            ${extra_arg}
            -O3
            -target spirv -emit-spirv-directly -source-embed-style text -profile sm_6_6+spirv_1_6
            -o  "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.hpp"
            -source-embed-name "shader_${stem}"
            ${abs_path}
        DEPENDS ${abs_path}
    )

    add_custom_command(
            OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.cuh"
            COMMAND
                ${SLANG_COMPILER_PATH}
                ${extra_arg} -DTARGET_CUDA=1
                -O3
                -target cuda
                -o "${CMAKE_CURRENT_BINARY_DIR}/shader/tmp_${stem}.cuh"
                ${abs_path}
            COMMAND
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/fix_generated_cuda.py
                "${CMAKE_CURRENT_BINARY_DIR}/shader/tmp_${stem}.cuh"
            DEPENDS
                ${abs_path} ${infile}
                ${CMAKE_CURRENT_SOURCE_DIR}/fix_generated_cuda.py
            COMMENT "Generating CUDA for ${stem}"
    )
endfunction()

if (${PPB_ENABLE_Slang_Vulkan} OR ${PPB_ENABLE_Slang_Cuda})
    slang_compile_shader(shader/eval.slang)
endif()



# VULKAN
if (${PPB_ENABLE_Slang_Vulkan})
    add_library(polyhedral_slang_vulkan_lib STATIC Impl_Slang_Vulkan.cpp "${CMAKE_CURRENT_BINARY_DIR}/shader/eval.hpp")
    target_compile_features(polyhedral_slang_vulkan_lib PUBLIC cxx_std_17)
    target_link_libraries(polyhedral_slang_vulkan_lib PUBLIC polyhedral_lib kompute::kompute glm::glm common_vulkan_lib)
    target_include_directories(polyhedral_slang_vulkan_lib PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

    add_executable(polyhedral_slang_vulkan ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
    target_link_libraries(polyhedral_slang_vulkan PRIVATE polyhedral_slang_vulkan_lib)
endif ()
# CUDA
if (${PPB_ENABLE_Slang_Cuda})
    add_library(polyhedral_slang_cuda_lib STATIC Impl_Slang_Cuda.cu wrapper_eval.cu "${CMAKE_CURRENT_BINARY_DIR}/shader/eval.cuh")
    target_include_directories(polyhedral_slang_cuda_lib PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(polyhedral_slang_cuda_lib PUBLIC polyhedral_lib)

    add_executable(polyhedral_slang_cuda ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
    target_link_libraries(polyhedral_slang_cuda PRIVATE polyhedral_slang_cuda_lib)
endif ()
