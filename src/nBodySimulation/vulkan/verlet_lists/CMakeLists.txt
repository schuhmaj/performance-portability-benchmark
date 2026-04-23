vulkan_compile_shaders_path(${CMAKE_CURRENT_SOURCE_DIR}/shaders HEADERS ppb)

add_library(nbody_vulkan_verlet_lib STATIC
        Impl_Vulkan.h Impl_Vulkan.cpp
        ${HEADERS}
)

target_compile_features(nbody_vulkan_verlet_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_vulkan_verlet_lib PUBLIC nbody_lib kompute::kompute common_vulkan_lib)
target_include_directories(nbody_vulkan_verlet_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

add_executable(nbody_vulkan_verlet main.cpp)
target_link_libraries(nbody_vulkan_verlet PRIVATE nbody_vulkan_verlet_lib)