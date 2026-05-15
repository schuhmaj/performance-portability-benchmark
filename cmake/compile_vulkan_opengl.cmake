# Find the GLSL Compiler
find_program(GLS_LANG_COMPILER_PATH NAMES glslc REQUIRED)

# - Provides a function to compile GLSL/OpenGL compute shader (*.comp) files to C++ header files.
# Usage:
#   compile_vulkan(INFILE <input_file> [<input_file> ...]
#                         PATH <output_dir>
#                         OUT_HEADERS <output_variable>
#                         [KERNEL_NAME <variable_name> [<variable_name> ...]]
#                         [DEFINITIONS <definition> ...]
#                         [NAMESPACE <namespace_name>]
#   )
#
# LANGUAGE currently only supports SPIR-V.
#
# KERNEL_NAME is optional. If omitted, the input file stem is used as the generated
# C++ variable name. It can be given once for all input files, or once per input file.
#
# DEFINITIONS is optional and is passed directly to glslc, e.g.
#   -DFloatType=float -DFloatType3=vec3
#
# NAMESPACE is optional and defaults to "ppb".
#
# Produces one header file per input:
#   ${PATH}/<input_file_stem>.h
#
# The generated files are stored in <output_variable> and marked as GENERATED.
function(compile_vulkan_opengl)
    set(options)
    set(oneValueArgs PATH OUT_HEADERS NAMESPACE)
    set(multiValueArgs INFILE KERNEL_NAME DEFINITIONS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT ARG_INFILE)
        message(FATAL_ERROR "compile_vulkan: INFILE is required")
    endif ()
    if (NOT ARG_PATH)
        message(FATAL_ERROR "compile_vulkan: PATH is required")
    endif ()
    if (NOT ARG_OUT_HEADERS)
        message(FATAL_ERROR "compile_vulkan: OUT_HEADERS is required")
    endif ()

    # Validate KERNEL_NAME list length: must be empty, 1, or equal to number of INFILEs.
    list(LENGTH ARG_INFILE _infile_count)
    list(LENGTH ARG_KERNEL_NAME _kernel_count)
    if (_kernel_count GREATER 0
            AND NOT _kernel_count EQUAL 1
            AND NOT _kernel_count EQUAL _infile_count)
        message(FATAL_ERROR
                "vulkan_compile_opengl: KERNEL_NAME must be either omitted, a single value, "
                "or a list with the same length as INFILE "
                "(got ${_kernel_count} kernel names for ${_infile_count} input files).")
    endif ()

    set(generated_headers "")
    set(_index 0)

    foreach (infile IN LISTS ARG_INFILE)
        cmake_path(GET infile STEM LAST_ONLY stem)

        set(out_file "${ARG_PATH}/${stem}.h")
        set(tmp_num "${ARG_PATH}/${stem}.num")

        if (NOT IS_ABSOLUTE "${infile}")
            set(abs_path "${CMAKE_CURRENT_SOURCE_DIR}/${infile}")
        else ()
            set(abs_path "${infile}")
        endif ()

        if (_kernel_count EQUAL 0)
            set(KERNEL_NAME "${stem}")
        elseif (_kernel_count EQUAL 1)
            list(GET ARG_KERNEL_NAME 0 KERNEL_NAME)
        else ()
            list(GET ARG_KERNEL_NAME ${_index} KERNEL_NAME)
        endif ()

        add_custom_command(
                OUTPUT "${out_file}" "${tmp_num}"
                COMMAND ${GLS_LANG_COMPILER_PATH}
                ${ARG_DEFINITIONS}
                -mfmt=num
                -O
                -o "${tmp_num}"
                "${abs_path}"
                COMMAND ${CMAKE_COMMAND}
                -D "INPUT_FILE=${tmp_num}"
                -D "OUTPUT_FILE=${out_file}"
                -D "VAR_NAME=${KERNEL_NAME}"
                -D "NAMESPACE=${ARG_NAMESPACE}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/scripts/spirv_into_header.cmake"
                COMMAND_EXPAND_LISTS
                DEPENDS "${abs_path}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/scripts/spirv_into_header.cmake"
                COMMENT "GLSL -> SPIR-V: ${stem}.h"
                VERBATIM
        )

        list(APPEND generated_headers "${out_file}")
        math(EXPR _index "${_index} + 1")
    endforeach ()

    set_source_files_properties(${generated_headers}
            PROPERTIES GENERATED TRUE
    )
    set(${ARG_OUT_HEADERS} "${generated_headers}" PARENT_SCOPE)
endfunction()