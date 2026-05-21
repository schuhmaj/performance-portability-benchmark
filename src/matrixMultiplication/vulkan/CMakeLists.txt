compile_vulkan_opengl(
        INFILE MatrixMultiplicationShader.comp
        KERNEL_NAME matmul_kernel
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        OUT_HEADERS OPENGL_HEADERS
)

add_library(matMul_vulkan_lib STATIC Impl_Vulkan.h Impl_Vulkan.cpp ${OPENGL_HEADERS})
target_compile_features(matMul_vulkan_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_vulkan_lib PUBLIC matMul_lib kompute::kompute common_vulkan_lib)
target_include_directories(matMul_vulkan_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

add_executable(matMul_vulkan main.cpp)
target_link_libraries(matMul_vulkan PRIVATE matMul_vulkan_lib)