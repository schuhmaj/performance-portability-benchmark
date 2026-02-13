# function for compiling slang shaders to vulkan
function(slang_vulkan_compile_shaders KERNEL TARGET_NAME NAMESPACE)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders)

    set(COMP_FILE ${KERNEL}.comp)
    set(SLANG_FILE ${KERNEL}.slang)
    set(OUT_HEADER ${KERNEL}.h)

    add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${COMP_FILE}
    COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_FILE} 
            -profile glsl_450 # compiling to OpenGL 4.5 to keep shader targets consistent with direct Vulkan implementations
            -target glsl 
            -o ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${COMP_FILE}
            -entry computeMain
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SLANG_FILE}
    COMMENT "Slang -> GLSL (.comp): ${KERNEL}"
    )
    add_custom_target(generate_${TARGET_NAME}_${COMP_FILE} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generated_shaders/${COMP_FILE})

    vulkan_compile_shader(
        INFILE generated_shaders/${COMP_FILE}
        OUTFILE ${OUT_HEADER}
        NAMESPACE ${NAMESPACE}
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
    )
    
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${OUT_HEADER}
        PROPERTIES GENERATED TRUE
    )

endfunction()

# function for compiling slang shaders to ptx
function(slang_ptx_compile_shaders KERNEL TARGET_NAME)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated_shaders)

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

endfunction()