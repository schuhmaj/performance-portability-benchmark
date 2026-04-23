# The CPP library can be linked to e.g. a test executable
add_library(polyhedral_cpp_lib STATIC Impl_Cpp.cpp)
target_compile_features(polyhedral_cpp_lib PUBLIC cxx_std_20)
target_link_libraries(polyhedral_cpp_lib
        PUBLIC
        polyhedral_lib
)

# This is the main executable to be used for benchmarking purposes
add_executable(polyhedral_cpp ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_features(polyhedral_cpp PUBLIC cxx_std_20)
target_link_libraries(polyhedral_cpp
        PRIVATE
        polyhedral_cpp_lib
)
