# - Provides a function to compile a OpenCL Kernel (*.cl) file to a C++ Header File (*.h)
# Usage:
#   compile_opencl(INFILE <input_file>
#                  OUTFILE <header_name.h>
#                  PATH <relative_output_dir>
#                  VAR_NAME <name_variable_header_file>
#                  NAMESPACE <namespace_name_variable>
#   )
#  VAR_NAME (defaults to "KERNEL_SOURCE") and NAMESPACE (defaults to "ppb") are optional arguments.
# Produces a file ${PATH}/${OUTFILE}
# with content:
#   #pragma once
#   namespace <NAMESPACE> {
#       inline constexpr const char <VAR_NAME>[] = R"( ... file contents ... )";
#   }
function(compile_opencl)
    set(options)
    set(oneValueArgs INFILE OUTFILE PATH VAR_NAME NAMESPACE)
    set(multiValueArgs)
    cmake_parse_arguments(COMPILE_OCL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT COMPILE_OCL_INFILE)
        message(FATAL_ERROR "compile_opencl: INFILE is required")
    endif ()
    if (NOT COMPILE_OCL_OUTFILE)
        message(FATAL_ERROR "compile_opencl: OUTFILE is required")
    endif ()
    if (NOT COMPILE_OCL_PATH)
        message(FATAL_ERROR "compile_opencl: PATH is required")
    endif ()

    if (NOT DEFINED COMPILE_OCL_VAR_NAME OR COMPILE_OCL_VAR_NAME STREQUAL "")
        set(COMPILE_OCL_VAR_NAME "KERNEL_SOURCE")
    endif ()
    if (NOT DEFINED COMPILE_OCL_NAMESPACE OR COMPILE_OCL_NAMESPACE STREQUAL "")
        set(COMPILE_OCL_NAMESPACE "ppb")
    endif ()

    set(OUTPUT_PATH "${COMPILE_OCL_PATH}/${COMPILE_OCL_OUTFILE}")

    # Make the input path absolute
    if(NOT IS_ABSOLUTE "${COMPILE_OCL_INFILE}")
        set(INPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${COMPILE_OCL_INFILE}")
    else()
        set(INPUT_PATH "${COMPILE_OCL_INFILE}")
    endif()

    # Make the output directory at configure time
    file(MAKE_DIRECTORY "${COMPILE_OCL_PATH}")

    # Create a custom command that runs at build time
    add_custom_command(
            OUTPUT "${OUTPUT_PATH}"
            COMMAND ${CMAKE_COMMAND}
            -D "INPUT_FILE=${INPUT_PATH}"
            -D "OUTPUT_FILE=${OUTPUT_PATH}"
            -D "VAR_NAME=${COMPILE_OCL_VAR_NAME}"
            -D "NAMESPACE=${COMPILE_OCL_NAMESPACE}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/opencl_to_header.cmake"
            DEPENDS "${INPUT_PATH}"
            COMMENT "Generating OpenCL header ${COMPILE_OCL_OUTFILE}"
            VERBATIM
    )
endfunction()