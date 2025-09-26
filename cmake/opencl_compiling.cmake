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

    # Read input file as a single string
    file(READ "${COMPILE_OCL_INFILE}" FILE_CONTENTS)

    set(RAW_DELIM "myDelimiter")
    file(MAKE_DIRECTORY "${COMPILE_OCL_PATH}")
    set(OUTPUT_PATH "${COMPILE_OCL_PATH}/${COMPILE_OCL_OUTFILE}")

    # Generate header content
    set(HEADER_TEXT
            "#pragma once

namespace ${COMPILE_OCL_NAMESPACE} {

inline constexpr const char ${COMPILE_OCL_VAR_NAME}[] = R\"${RAW_DELIM}(
${FILE_CONTENTS}
)${RAW_DELIM}\";

} // namespace ${COMPILE_OCL_NAMESPACE}
            ")
    # Write header
    file(WRITE "${OUTPUT_PATH}" "${HEADER_TEXT}")
endfunction()