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
vulkan_compile_shader(
        INFILE KernelHistogram.comp
        OUTFILE KernelHistogram.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
vulkan_compile_shader(
        INFILE KernelBlellochScan.comp
        OUTFILE KernelBlellochScan.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
vulkan_compile_shader(
        INFILE KernelBlockSum.comp
        OUTFILE KernelBlockSum.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
vulkan_compile_shader(
        INFILE KernelIdCells.comp
        OUTFILE KernelIdCells.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
add_library(nbody_vulkan_lib STATIC
        Impl_Vulkan.h Impl_Vulkan.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/KernelForce.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelVelocity.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelPosition.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelBlellochScan.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelBlockSum.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelHistogram.h"
        "${CMAKE_CURRENT_BINARY_DIR}/KernelIdCells.h"

)
target_compile_features(nbody_vulkan_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_vulkan_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
target_include_directories(nbody_vulkan_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

add_executable(nbody_vulkan main.cpp)
target_link_libraries(nbody_vulkan PRIVATE nbody_vulkan_lib)