set(SHADER_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelForce.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelVelocity.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelPosition.comp
)

compile_vulkan_opengl(
        INFILE ${SHADER_SOURCE_FILES}
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        OUT_HEADERS OPENGL_HEADERS
)

add_library(nbody_vulkan_naive_lib STATIC
        Impl_Vulkan.h Impl_Vulkan.cpp
        ${OPENGL_HEADERS}
)

target_compile_features(nbody_vulkan_naive_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_vulkan_naive_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
target_include_directories(nbody_vulkan_naive_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

add_executable(nbody_vulkan_naive main.cpp)
target_link_libraries(nbody_vulkan_naive PRIVATE nbody_vulkan_naive_lib)