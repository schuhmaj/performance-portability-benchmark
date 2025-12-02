vulkan_compile_shader(
        INFILE KernelPosition.comp
        OUTFILE KernelPosition.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
vulkan_compile_shader(
        INFILE KernelForce.comp
        OUTFILE KernelForce.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
vulkan_compile_shader(
        INFILE KernelVelocity.comp
        OUTFILE KernelVelocity.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
add_library(nbody_vulkan_naive_lib STATIC
        Impl_Vulkan.h Impl_Vulkan.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/KernelForce.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelVelocity.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelPosition.h"
)
target_compile_features(nbody_vulkan_naive_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_vulkan_naive_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
target_include_directories(nbody_vulkan_naive_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

add_executable(nbody_vulkan_naive main.cpp)
target_link_libraries(nbody_vulkan_naive PRIVATE nbody_vulkan_naive_lib)