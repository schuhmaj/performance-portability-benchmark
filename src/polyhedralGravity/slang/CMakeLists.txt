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

slang_compile_shader(shader/eval.slang)

add_library(compute_shader_slang INTERFACE
    "${CMAKE_CURRENT_BINARY_DIR}/shader/eval.hpp"
)

add_library(cuda_slang INTERFACE
    "${CMAKE_CURRENT_BINARY_DIR}/shader/eval.cuh"
)

find_package(Vulkan REQUIRED)
add_exe(slang_vulkan LIBS compute_shader_slang glm::glm Vulkan::Vulkan ${CMAKE_DL_LIBS})
target_include_directories(lib_slang_vulkan PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

find_package(CUDAToolkit REQUIRED)

add_library(slang_cuda_wrapper STATIC wrapper_eval.cu)
target_link_libraries(slang_cuda_wrapper PUBLIC cuda_slang CUDA::cudart)
target_compile_definitions(slang_cuda_wrapper PUBLIC -DFLOAT_BITS=${FLOAT_BITS})
target_include_directories(slang_cuda_wrapper PUBLIC "${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_SOURCE_DIR}/inc)

add_exe(slang_cuda EXT cu  LIBS cuda_slang slang_cuda_wrapper glm::glm CUDA::cudart)
target_include_directories(lib_slang_cuda PUBLIC "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/implementations/cuda/inc")
