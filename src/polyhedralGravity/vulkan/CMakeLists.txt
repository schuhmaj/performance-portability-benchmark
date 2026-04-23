vulkan_compile_opengl(
        INFILE shader/init.comp
        OUTFILE vulkan_init.h
        VAR_NAME INIT_SPRIV
        PATH ${CMAKE_CURRENT_BINARY_DIR}/shader
)
vulkan_compile_opengl(
        INFILE shader/eval.comp
        OUTFILE vulkan_eval.h
        VAR_NAME EVAL_SPIRV
        PATH ${CMAKE_CURRENT_BINARY_DIR}/shader
)

add_library(polyhedral_vulkan_lib STATIC Impl_Vulkan.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/shader/vulkan_init.h
        ${CMAKE_CURRENT_BINARY_DIR}/shader/vulkan_eval.h
)
target_compile_features(polyhedral_vulkan_lib PUBLIC cxx_std_17)
target_link_libraries(polyhedral_vulkan_lib PUBLIC polyhedral_lib kompute::kompute glm::glm common_vulkan_lib)
target_include_directories(polyhedral_vulkan_lib PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/shader
)

add_executable(polyhedral_vulkan ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_link_libraries(polyhedral_vulkan PRIVATE polyhedral_vulkan_lib)
