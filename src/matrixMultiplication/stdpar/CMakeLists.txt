# The Stdpar library can be linked to e.g. a test executable
add_library(matMul_stdpar_lib STATIC Impl_Stdpar.h Impl_Stdpar.cpp)
target_compile_features(matMul_stdpar_lib PUBLIC cxx_std_20)
target_link_libraries(matMul_stdpar_lib
        PUBLIC
        matMul_lib
        ppb_stdpar_offload
)

# This is the main executable to be used for benchmarking purposes
add_executable(matMul_stdpar main.cpp)
target_link_libraries(matMul_stdpar PUBLIC matMul_stdpar_lib)
