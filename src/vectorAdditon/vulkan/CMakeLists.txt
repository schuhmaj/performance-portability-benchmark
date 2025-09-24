vulkan_compile_shader(
        INFILE VectorAdditionShader.comp
        OUTFILE VectorAdditionShader.h
        NAMESPACE "ppb"
        RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
)
add_executable(vec_vulkan Impl_Vulkan.cpp "${CMAKE_CURRENT_BINARY_DIR}/VectorAdditionShader.h")
target_compile_features(vec_vulkan PUBLIC cxx_std_17)
target_link_libraries(vec_vulkan PUBLIC vec_lib kompute::kompute)
target_include_directories(vec_vulkan PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)