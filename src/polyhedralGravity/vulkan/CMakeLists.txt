if (${PPB_Polyhedral_FloatType} EQUAL 32)
    set(OPENGL_COMPILER_DEFINITIONS -DFloatType=float -DFloatType3=vec3 -DFloatType4=vec4 -DFloatTypeM=mat3)
elseif (${PPB_Polyhedral_FloatType} EQUAL 64)
    set(OPENGL_COMPILER_DEFINITIONS -DFloatType=double -DFloatType3=dvec3 -DFloatType4=dvec4 -DFloatTypeM=dmat3)
endif ()

compile_vulkan_opengl(
        INFILE ${CMAKE_CURRENT_SOURCE_DIR}/shader/init.comp ${CMAKE_CURRENT_SOURCE_DIR}/shader/eval.comp
        KERNEL_NAME INIT_SPRIV EVAL_SPIRV
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        OUT_HEADERS OPENGL_HEADERS
        DEFINITIONS ${OPENGL_COMPILER_DEFINITIONS}
)

add_library(polyhedral_vulkan_lib STATIC Impl_Vulkan.cpp ${OPENGL_HEADERS})
target_compile_features(polyhedral_vulkan_lib PUBLIC cxx_std_17)
target_link_libraries(polyhedral_vulkan_lib PUBLIC polyhedral_lib kompute::kompute glm::glm common_vulkan_lib)
target_include_directories(polyhedral_vulkan_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_executable(polyhedral_vulkan ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_vulkan PRIVATE PARADIGM="Vulkan")
target_link_libraries(polyhedral_vulkan PRIVATE polyhedral_vulkan_lib)
