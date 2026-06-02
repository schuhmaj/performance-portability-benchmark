compile_vulkan_opengl(
        INFILE ${CMAKE_CURRENT_SOURCE_DIR}/VectorAdditionShader.comp
        KERNEL_NAME vector_addition
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        OUT_HEADERS OPENGL_HEADERS
        DEFINITIONS ${OPENGL_COMPILER_DEFINITIONS}
)

add_executable(vec_vulkan Impl_Vulkan.cpp ${OPENGL_HEADERS})
target_compile_features(vec_vulkan PUBLIC cxx_std_17)
target_link_libraries(vec_vulkan PUBLIC vec_lib kompute::kompute common_vulkan_lib)
target_include_directories(vec_vulkan PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)