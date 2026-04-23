function(vulkan_compile_shader infile)
    find_program(GLS_LANG_COMPILER_PATH NAMES glslc)

    cmake_path(GET infile PARENT_PATH parent)
    cmake_path(GET infile STEM LAST_ONLY stem)

    cmake_path(ABSOLUTE_PATH infile OUTPUT_VARIABLE abs_path)

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shader")

    set(extra_arg "")
    if (FLOAT_BITS EQUAL 32)
        set(extra_arg -DFloatType=float -DFloatType3=vec3 -DFloatType4=vec4 -DFloatTypeM=mat3)
    endif()

    add_custom_command(
        OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.hpp"
        COMMAND
            ${GLS_LANG_COMPILER_PATH}
            ${extra_arg}
            -mfmt=c
            -O
            -o  "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.hpp"
            ${abs_path}
        DEPENDS ${abs_path}
    )
endfunction()

find_package(Vulkan REQUIRED)

vulkan_compile_shader(shader/init.comp)
vulkan_compile_shader(shader/eval.comp)

add_library(compute_shader INTERFACE
    "${CMAKE_CURRENT_BINARY_DIR}/shader/init.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/shader/eval.hpp"
)

include(glm)
add_exe(opengl_vulkan LIBS glm::glm compute_shader Vulkan::Vulkan ${CMAKE_DL_LIBS})
target_include_directories(lib_opengl_vulkan PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
