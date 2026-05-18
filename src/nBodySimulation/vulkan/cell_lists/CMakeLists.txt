set(OPENGL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelForce.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelVelocity.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelPosition.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelBlellochScan.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelBlockSum.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelIdCells.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelResetCells.comp
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/KernelHistogram.comp
)

compile_vulkan_opengl(
        INFILE ${OPENGL_SOURCE_FILES}
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        OUT_HEADERS OPENGL_HEADERS
)

add_library(nbody_vulkan_cells_lib STATIC
        Impl_Vulkan.h Impl_Vulkan.cpp
        ${OPENGL_HEADERS}
)

target_compile_features(nbody_vulkan_cells_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_vulkan_cells_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
target_include_directories(nbody_vulkan_cells_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

add_executable(nbody_vulkan_cells main.cpp)
target_link_libraries(nbody_vulkan_cells PRIVATE nbody_vulkan_cells_lib)