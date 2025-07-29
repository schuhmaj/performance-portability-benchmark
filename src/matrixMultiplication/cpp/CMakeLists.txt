# The CPP library can be linked to e.g. a test executable
add_library(matMul_cpp_lib STATIC Impl_Cpp.h Impl_Cpp.cpp)
target_compile_features(matMul_cpp_lib PUBLIC cxx_std_20)
target_link_libraries(matMul_cpp_lib
        PUBLIC
        matMul_lib
)

# This is the main executable to be used for benchmarking purposes
add_executable(matMul_cpp main.cpp)
target_compile_features(matMul_cpp PUBLIC cxx_std_20)
target_link_libraries(matMul_cpp
        PRIVATE
        matMul_cpp_lib
)
