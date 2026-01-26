function(slang_vulkan_compile_shader)
    set(oneValueArgs SLANG_FILE ENTRY NAMESPACE OUT_HEADER)
    cmake_parse_arguments(SLANG "" "${oneValueArgs}" "" ${ARGN})

    get_filename_component(stem ${SLANG_SLANG_FILE} NAME_WE)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${stem}.comp 
        COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_SLANG_FILE} 
            -profile glsl_450 
            -target glsl 
            -o ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${stem}.comp 
            -entry ${SLANG_ENTRY}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_SLANG_FILE}
        COMMENT "Slang -> GLSL (.comp): ${stem}"
    )
    add_custom_target(generate_naive_${stem}.comp DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${stem}.comp)

    vulkan_compile_shader(
        INFILE generated_shaders/${stem}.comp 
        OUTFILE ${SLANG_OUT_HEADER}
        NAMESPACE "${SLANG_NAMESPACE}"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
    )

    set_source_files_properties(${header_file}
        PROPERTIES GENERATED TRUE
    )
endfunction()

if (${PPB_ENABLE_Slang_Vulkan})
    slang_vulkan_compile_shader(
        SLANG_FILE KernelPosition.slang
        ENTRY computePosition
        NAMESPACE ppb
        OUT_HEADER KernelPosition.h
    )

    slang_vulkan_compile_shader(
        SLANG_FILE KernelVelocity.slang
        ENTRY computeVelocity
        NAMESPACE ppb
        OUT_HEADER KernelVelocity.h
    )

    slang_vulkan_compile_shader(
        SLANG_FILE KernelForce.slang
        ENTRY computeForce
        NAMESPACE ppb
        OUT_HEADER KernelForce.h
    )

    add_library(nbody_slang_vulkan_naive_lib STATIC
        Impl_Slang_Vulkan.h Impl_Slang_Vulkan.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/KernelForce.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelVelocity.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelPosition.h"
    )

    target_compile_features(nbody_slang_vulkan_naive_lib PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_vulkan_naive_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
    target_include_directories(nbody_slang_vulkan_naive_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

    add_executable(nbody_slang_vulkan_naive main.cpp)
    target_link_libraries(nbody_slang_vulkan_naive PRIVATE nbody_slang_vulkan_naive_lib)
endif()

function(slang_cuda_compile_shader)
    set(oneValueArgs SLANG_FILE ENTRY OUT_HEADER PARAMS)
    cmake_parse_arguments(SLANG "" "${oneValueArgs}" "" ${ARGN})

    get_filename_component(stem ${SLANG_SLANG_FILE} NAME_WE)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${SLANG_OUT_HEADER}
        COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_SLANG_FILE} 
            -target ptx
            -entry ${SLANG_ENTRY}
            -o ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${SLANG_OUT_HEADER}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_SLANG_FILE}
        COMMAND python 
            ${CMAKE_CURRENT_SOURCE_DIR}/../rename_params.py 
            ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${SLANG_OUT_HEADER} 
            ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${SLANG_OUT_HEADER} 
            SLANG_globalParams 
            ${SLANG_PARAMS}
        COMMENT "Slang -> CUDA header (.ptx): ${stem}"
    )

    add_custom_target(generate_naive_${SLANG_OUT_HEADER} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${SLANG_OUT_HEADER})

    set(${SLANG_OUT_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${SLANG_OUT_HEADER} PARENT_SCOPE)
endfunction()

if (${PPB_ENABLE_Slang_Cuda})
    slang_cuda_compile_shader(
        SLANG_FILE KernelPosition.slang
        ENTRY computePosition
        OUT_HEADER KernelPosition.ptx
        PARAMS Params_Position
    )

    slang_cuda_compile_shader(
        SLANG_FILE KernelVelocity.slang
        ENTRY computeVelocity
        OUT_HEADER KernelVelocity.ptx
        PARAMS Params_Velocity
    )

    slang_cuda_compile_shader(
        SLANG_FILE KernelForce.slang
        ENTRY computeForce
        OUT_HEADER KernelForce.ptx
        PARAMS Params_Force
    )

    add_library(nbody_slang_cuda_naive_lib STATIC
        Impl_Slang_Cuda.cuh Impl_Slang_Cuda.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelPosition.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelVelocity.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelForce.ptx
    )
    target_compile_features(nbody_slang_cuda_naive_lib PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_cuda_naive_lib PUBLIC nbody_lib CUDA::cudart CUDA::cuda_driver)
    target_compile_options(nbody_slang_cuda_naive_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    set_target_properties(nbody_slang_cuda_naive_lib PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )

    add_executable(nbody_slang_cuda_naive main.cu)
    target_compile_features(nbody_slang_cuda_naive PUBLIC cxx_std_20)
    target_link_libraries(nbody_slang_cuda_naive PRIVATE nbody_slang_cuda_naive_lib)
    set_target_properties(nbody_slang_cuda_naive PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
    )
endif()
