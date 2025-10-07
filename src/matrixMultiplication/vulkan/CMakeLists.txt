vulkan_compile_shader(
        INFILE MatrixMultiplicationShader.comp
        OUTFILE MatrixMultiplicationShader.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
add_library(matMul_vulkan_lib STATIC Impl_Vulkan.h Impl_Vulkan.cpp "${CMAKE_CURRENT_BINARY_DIR}/MatrixMultiplicationShader.h")
target_compile_features(matMul_vulkan_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_vulkan_lib PUBLIC vec_lib kompute::kompute common_vulkan_lib)
target_include_directories(matMul_vulkan_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

add_executable(matMul_vulkan main.cpp)
target_link_libraries(matMul_vulkan PRIVATE matMul_vulkan_lib)