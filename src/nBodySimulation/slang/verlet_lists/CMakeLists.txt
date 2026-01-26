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
        add_custom_target(generate_verlet_lists_${stem}.comp DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${stem}.comp)

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
                SLANG_FILE KernelForce.slang
                ENTRY computeForce
                NAMESPACE ppb
                OUT_HEADER KernelForce.h
        )
        slang_vulkan_compile_shader(
                SLANG_FILE KernelVelocity.slang
                ENTRY computeVelocity
                NAMESPACE ppb
                OUT_HEADER KernelVelocity.h
        )
        slang_vulkan_compile_shader(
                SLANG_FILE KernelBlellochScan.slang
                ENTRY computeBlellochScan
                NAMESPACE ppb
                OUT_HEADER KernelBlellochScan.h
        )
        slang_vulkan_compile_shader(
                SLANG_FILE KernelBlockSum.slang
                ENTRY computeBlockSum
                NAMESPACE ppb
                OUT_HEADER KernelBlockSum.h
        )
        slang_vulkan_compile_shader(
                SLANG_FILE KernelCountNeighbors.slang
                ENTRY computeCountNeighbors
                NAMESPACE ppb
                OUT_HEADER KernelCountNeighbors.h
        )
        slang_vulkan_compile_shader(
                SLANG_FILE KernelVerlet.slang
                ENTRY computeVerlet
                NAMESPACE ppb
                OUT_HEADER KernelVerlet.h
        )
        add_library(nbody_slang_vulkan_verlet_lib STATIC
                Impl_Slang_Vulkan.h Impl_Slang_Vulkan.cpp
                "${CMAKE_CURRENT_BINARY_DIR}/KernelForce.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelVelocity.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelPosition.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelBlellochScan.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelBlockSum.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelCountNeighbors.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelVerlet.h"
        )

        target_compile_features(nbody_slang_vulkan_verlet_lib PUBLIC cxx_std_20)
        target_link_libraries(nbody_slang_vulkan_verlet_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
        target_include_directories(nbody_slang_vulkan_verlet_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

        add_executable(nbody_slang_vulkan_verlet main.cpp)
        target_link_libraries(nbody_slang_vulkan_verlet PRIVATE nbody_slang_vulkan_verlet_lib)
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
                OUT_HEADER KernelPosition.h
                PARAMS Params_Position
        )
        slang_cuda_compile_shader(
                SLANG_FILE KernelForce.slang
                ENTRY computeForce
                OUT_HEADER KernelForce.h
                PARAMS Params_Force
        )
        slang_cuda_compile_shader(
                SLANG_FILE KernelVelocity.slang
                ENTRY computeVelocity
                OUT_HEADER KernelVelocity.h
                PARAMS Params_Velocity
        )
        slang_cuda_compile_shader(
                SLANG_FILE KernelBlellochScan.slang
                ENTRY computeBlellochScan
                OUT_HEADER KernelBlellochScan.h
                PARAMS Params_BlellochScan
        )
        slang_cuda_compile_shader(
                SLANG_FILE KernelBlockSum.slang
                ENTRY computeBlockSum
                OUT_HEADER KernelBlockSum.h
                PARAMS Params_BlockSum
        )
        slang_cuda_compile_shader(
                SLANG_FILE KernelCountNeighbors.slang
                ENTRY computeCountNeighbors
                OUT_HEADER KernelCountNeighbors.h
                PARAMS Params_CountNeighbors
        )
        slang_cuda_compile_shader(
                SLANG_FILE KernelVerlet.slang
                ENTRY computeVerlet
                OUT_HEADER KernelVerlet.h
                PARAMS Params_Verlet
        )

        add_library(nbody_slang_cuda_verlet_lib STATIC
        Impl_Slang_Cuda.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelPosition.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelVelocity.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelForce.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelBlellochScan.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelBlockSum.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelCountNeighbors.ptx
        ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/KernelVerlet.ptx
        )
        target_compile_features(nbody_slang_cuda_verlet_lib PUBLIC cxx_std_20)
        target_link_libraries(nbody_slang_cuda_verlet_lib PUBLIC nbody_lib)
        target_compile_options(nbody_slang_cuda_verlet_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
        set_target_properties(nbody_slang_cuda_verlet_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
        )
        # target_include_directories(nbody_slang_cuda_verlet_lib PUBLIC
        #     ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders
        # )

        add_executable(nbody_slang_cuda_verlet main.cpp)
        target_link_libraries(nbody_slang_cuda_verlet PRIVATE nbody_slang_cuda_verlet_lib)
endif()
