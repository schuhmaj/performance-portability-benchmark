# The CPP library can be linked to e.g. a test executable
add_library(nbody_cpp_lib STATIC Impl_Cpp.h Impl_Cpp.cpp)
target_compile_features(nbody_cpp_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_cpp_lib
    PUBLIC
        nbody_lib
)

# This is the main executable to be used for benchmarking purposes
add_executable(nbody_cpp main.cpp)
target_compile_features(nbody_cpp PUBLIC cxx_std_20)
target_link_libraries(nbody_cpp
    PRIVATE
        nbody_cpp_lib
)
