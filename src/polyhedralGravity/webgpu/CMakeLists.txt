function(compile_webgpu infile)
    cmake_path(GET infile PARENT_PATH parent)
    cmake_path(GET infile STEM LAST_ONLY stem)

    cmake_path(ABSOLUTE_PATH infile OUTPUT_VARIABLE abs_path)

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shader")

    add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.hpp"
            COMMAND
            ${Python_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/util/create_header.py
            ${abs_path}  "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.hpp"
            DEPENDS ${abs_path}
    )
endfunction()

compile_webgpu("shader/shader_init.wgsl")
compile_webgpu("shader/shader_eval.wgsl")
add_library(webgpu_shader INTERFACE
        "${CMAKE_CURRENT_BINARY_DIR}/shader/shader_init.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/shader/shader_eval.hpp"
)


find_package(WGPU REQUIRED)
add_exe(webgpu LIBS webgpu_shader FORCE_BITS 32)
target_link_libraries(lib_webgpu PUBLIC wgpu::wgpu)
target_include_directories(lib_webgpu PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/shader")
