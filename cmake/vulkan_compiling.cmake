function(vulkan_compile_opengl)
    set(options)
    set(oneValueArgs INFILE OUTFILE PATH VAR_NAME NAMESPACE)
    set(multiValueArgs)
    cmake_parse_arguments(COMPILE_VULKAN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT COMPILE_VULKAN_INFILE)
        message(FATAL_ERROR "compile_opencl: INFILE is required")
    endif ()
    if (NOT COMPILE_VULKAN_OUTFILE)
        message(FATAL_ERROR "compile_opencl: OUTFILE is required")
    endif ()
    if (NOT COMPILE_VULKAN_PATH)
        message(FATAL_ERROR "compile_opencl: PATH is required")
    endif ()

    if (NOT DEFINED COMPILE_VULKAN_VAR_NAME OR COMPILE_VULKAN_VAR_NAME STREQUAL "")
        set(COMPILE_VULKAN_VAR_NAME "KERNEL_SPIRV")
    endif ()
    if (NOT DEFINED COMPILE_VULKAN_NAMESPACE OR COMPILE_VULKAN_NAMESPACE STREQUAL "")
        set(COMPILE_VULKAN_NAMESPACE "ppb")
    endif ()

    set(OUTPUT_PATH "${COMPILE_VULKAN_PATH}/${COMPILE_VULKAN_OUTFILE}")

    if(NOT IS_ABSOLUTE "${COMPILE_VULKAN_INFILE}")
        set(INPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${COMPILE_VULKAN_INFILE}")
    else()
        set(INPUT_PATH "${COMPILE_VULKAN_INFILE}")
    endif()
    cmake_path(GET COMPILE_VULKAN_INFILE STEM LAST_ONLY stem)

    set(extra_arg "")
    if (FLOAT_BITS EQUAL 32)
        set(extra_arg -DFloatType=float -DFloatType3=vec3 -DFloatType4=vec4 -DFloatTypeM=mat3)
    endif()

    find_program(GLS_LANG_COMPILER_PATH NAMES glslc REQUIRED)


    set(tmp_num "${CMAKE_CURRENT_BINARY_DIR}/shader/${stem}.num")

    # 1. Compile to list of comma-separated hex numbers
    add_custom_command(
            OUTPUT "${tmp_num}"
            COMMAND ${GLS_LANG_COMPILER_PATH}
            ${extra_arg}
            -mfmt=num
            -O
            -o "${tmp_num}"
            "${INPUT_PATH}"
            DEPENDS "${INPUT_PATH}"
            COMMENT "Compiling Vulkan shader ${COMPILE_VULKAN_VAR_NAME} to SPIR-V"
            VERBATIM
    )

    # 2. Wrap the hex numbers into a C++ header file
    add_custom_command(
            OUTPUT "${OUTPUT_PATH}"
            COMMAND ${CMAKE_COMMAND}
            -D "INPUT_FILE=${tmp_num}"
            -D "OUTPUT_FILE=${OUTPUT_PATH}"
            -D "VAR_NAME=${COMPILE_VULKAN_VAR_NAME}"
            -D "NAMESPACE=${COMPILE_VULKAN_NAMESPACE}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/scripts/vulkan_to_header.cmake"
            DEPENDS "${tmp_num}"
            COMMENT "Generating Vulkan header ${OUTPUT_PATH}"
            VERBATIM
    )
endfunction()