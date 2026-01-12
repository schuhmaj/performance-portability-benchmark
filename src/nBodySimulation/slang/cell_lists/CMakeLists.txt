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
    add_custom_target(generate_${stem} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${stem}.comp)

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

slang_vulkan_compile_shader(
        SLANG_FILE KernelPosition.slang
        ENTRY computeMain
        NAMESPACE ppb
        OUT_HEADER KernelPosition.h
)
slang_vulkan_compile_shader(
        SLANG_FILE KernelForce.slang
        ENTRY computeMain
        NAMESPACE ppb
        OUT_HEADER KernelForce.h
)
slang_vulkan_compile_shader(
        SLANG_FILE KernelVelocity.slang
        ENTRY computeMain
        NAMESPACE ppb
        OUT_HEADER KernelVelocity.h
)
slang_vulkan_compile_shader(
        SLANG_FILE KernelHistogram.slang
        ENTRY computeMain
        NAMESPACE ppb
        OUT_HEADER KernelHistogram.h
)
slang_vulkan_compile_shader(
        SLANG_FILE KernelBlellochScan.slang
        ENTRY computeMain
        NAMESPACE ppb
        OUT_HEADER KernelBlellochScan.h
)
slang_vulkan_compile_shader(
        SLANG_FILE KernelBlockSum.slang
        ENTRY computeMain
        NAMESPACE ppb
        OUT_HEADER KernelBlockSum.h
)
slang_vulkan_compile_shader(
        SLANG_FILE KernelIdCells.slang
        ENTRY computeMain
        NAMESPACE ppb
        OUT_HEADER KernelIdCells.h
)
slang_vulkan_compile_shader(
        SLANG_FILE KernelResetCells.slang
        ENTRY computeMain
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