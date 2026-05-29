# Define the executable using your HIP source files
add_executable(vec_hip Impl_Hip.hip)

# Link against your base library
target_link_libraries(vec_hip PRIVATE vec_lib)
target_compile_options(vec_hip PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-arch=native>)
