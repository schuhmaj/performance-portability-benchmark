# function for compiling slang shaders in a directory to vulkan
# note
#  - all files in the specified directory are expected to be .slang files
function(slang_vulkan_compile_shaders DIR_PATH OUT_TARGETS OUT_HEADERS TARGET_NAME NAMESPACE)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders)
    file(GLOB files CONFIGURE_DEPENDS "${DIR_PATH}/*")
    set(TARGETS "")
    set(HEADERS "")

    foreach(file IN LISTS files)
        get_filename_component(kernel "${file}" NAME_WE)
        set(COMP_FILE ${kernel}.comp)
        set(SLANG_FILE ${kernel}.slang)
        set(OUT_HEADER ${kernel}.h)

        add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${COMP_FILE}
        COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_FILE} 
                -profile glsl_450 # compiling to OpenGL 4.5 to keep shader targets consistent with direct Vulkan implementations
                -target glsl 
                -o ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${COMP_FILE}
                -entry computeMain
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_FILE}
        COMMENT "Slang -> GLSL (.comp): ${kernel}"
        )
        add_custom_target(generate_${TARGET_NAME}_${COMP_FILE} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${COMP_FILE})
        list(APPEND TARGETS "generate_${TARGET_NAME}_${COMP_FILE}")

        vulkan_compile_shader(
            INFILE generated_shaders/${COMP_FILE}
            OUTFILE ${OUT_HEADER}
            NAMESPACE ${NAMESPACE}
            RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
        )
        list(APPEND HEADERS "${CMAKE_CURRENT_BINARY_DIR}/${OUT_HEADER}")
    endforeach()

    set(${OUT_TARGETS} "${TARGETS}" PARENT_SCOPE)
    set(${OUT_HEADERS} "${HEADERS}" PARENT_SCOPE)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${OUT_HEADER}
        PROPERTIES GENERATED TRUE
    )

endfunction()

# function for compiling slang shaders in a directory to ptx
# note
#  - all files in the specified directory are expected to be .slang files
function(slang_ptx_compile_shaders DIR_PATH OUT_TARGETS TARGET_NAME)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders)
    file(GLOB files CONFIGURE_DEPENDS "${DIR_PATH}/*")
    set(TARGETS "")

    foreach(file IN LISTS files)
        get_filename_component(kernel "${file}" NAME_WE)
        set(PTX_FILE ${kernel}.ptx)
        set(SLANG_FILE ${kernel}.slang)

        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders/${PTX_FILE}
            COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_FILE} 
                -target ptx
                -entry computeMain
                -o ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders/${PTX_FILE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_SLANG_FILE}
            COMMENT "Slang -> CUDA header (.ptx): ${stem}"
        )
        add_custom_target(generate_${TARGET_NAME}_${PTX_FILE} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders/${PTX_FILE})
        list(APPEND TARGETS "generate_${TARGET_NAME}_${PTX_FILE}")
    endforeach()

    set(${OUT_TARGETS} "${TARGETS}" PARENT_SCOPE)

endfunction()
