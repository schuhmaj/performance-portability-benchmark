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
        add_custom_target(generate_cells_${stem}.comp DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${stem}.comp)

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
                SLANG_FILE KernelHistogram.slang
                ENTRY computeHistogram
                NAMESPACE ppb
                OUT_HEADER KernelHistogram.h
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
                SLANG_FILE KernelIdCells.slang
                ENTRY computeIdCells
                NAMESPACE ppb
                OUT_HEADER KernelIdCells.h
        )
        slang_vulkan_compile_shader(
                SLANG_FILE KernelResetCells.slang
                ENTRY computeResetCells
                NAMESPACE ppb
                OUT_HEADER KernelResetCells.h
        )
        add_library(nbody_slang_vulkan_cells_lib STATIC
                Impl_Slang_Vulkan.h Impl_Slang_Vulkan.cpp
                "${CMAKE_CURRENT_BINARY_DIR}/KernelForce.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelVelocity.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelPosition.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelBlellochScan.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelBlockSum.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelHistogram.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelIdCells.h"
                "${CMAKE_CURRENT_BINARY_DIR}/KernelResetCells.h"
        )

        target_compile_features(nbody_slang_vulkan_cells_lib PUBLIC cxx_std_20)
        target_link_libraries(nbody_slang_vulkan_cells_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
        target_include_directories(nbody_slang_vulkan_cells_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

        add_executable(nbody_slang_vulkan_cells main.cpp)
        target_link_libraries(nbody_slang_vulkan_cells PRIVATE nbody_slang_vulkan_cells_lib)
endif ()


function(slang_cuda_compile_shader)
    set(oneValueArgs SLANG_FILE ENTRY OUT_HEADER PARAMS)
    cmake_parse_arguments(SLANG "" "${oneValueArgs}" "" ${ARGN})

    get_filename_component(stem ${SLANG_SLANG_FILE} NAME_WE)

    set(OUTPUT_FILE ${SLANG_GENERATED_DIR}/${SLANG_OUT_HEADER})

    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_SLANG_FILE} 
            -target ptx
            -entry ${SLANG_ENTRY}
            -o ${OUTPUT_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_SLANG_FILE}
        COMMAND python3 
            ${CMAKE_CURRENT_SOURCE_DIR}/../rename_params.py 
            ${OUTPUT_FILE} 
            ${OUTPUT_FILE} 
            SLANG_globalParams 
            ${SLANG_PARAMS}
        COMMENT "Slang -> CUDA header (.ptx): ${stem}"
    )

    add_custom_target(generate_cells_${SLANG_OUT_HEADER} DEPENDS ${OUTPUT_FILE})

    set(${SLANG_OUT_HEADER} ${OUTPUT_FILE} PARENT_SCOPE)
endfunction()

if (${PPB_ENABLE_Slang_Cuda})
    set(SLANG_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders)
    file(MAKE_DIRECTORY ${SLANG_GENERATED_DIR})

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
    slang_cuda_compile_shader(
        SLANG_FILE KernelHistogram.slang
        ENTRY computeHistogram
        OUT_HEADER KernelHistogram.ptx
        PARAMS Params_Histogram
    )
    slang_cuda_compile_shader(
        SLANG_FILE KernelBlellochScan.slang
        ENTRY computeBlellochScan
        OUT_HEADER KernelBlellochScan.ptx
        PARAMS Params_ExclusiveScan
    )
    slang_cuda_compile_shader(
        SLANG_FILE KernelBlockSum.slang
        ENTRY computeBlockSum
        OUT_HEADER KernelBlockSum.ptx
        PARAMS Params_ExclusiveScan
    )
    slang_cuda_compile_shader(
        SLANG_FILE KernelIdCells.slang
        ENTRY computeIdCells
        OUT_HEADER KernelIdCells.ptx
        PARAMS Params_IdCells
    )
    slang_cuda_compile_shader(
        SLANG_FILE KernelResetCells.slang
        ENTRY computeResetCells
        OUT_HEADER KernelResetCells.ptx
        PARAMS Params_ResetCells
    )
    add_library(nbody_slang_cuda_cells_lib STATIC
        Impl_Slang_Cuda.cuh Impl_Slang_Cuda.cu
    )
    add_dependencies(nbody_slang_cuda_cells_lib
        generate_cells_KernelPosition.ptx
        generate_cells_KernelVelocity.ptx
        generate_cells_KernelForce.ptx
        generate_cells_KernelHistogram.ptx
        generate_cells_KernelBlellochScan.ptx
        generate_cells_KernelBlockSum.ptx
        generate_cells_KernelIdCells.ptx
        generate_cells_KernelResetCells.ptx
    )
    target_compile_definitions(nbody_slang_cuda_cells_lib PRIVATE
        SLANG_PTX_DIR="${SLANG_GENERATED_DIR}"
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
