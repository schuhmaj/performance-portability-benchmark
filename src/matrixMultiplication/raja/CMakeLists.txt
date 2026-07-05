ppb_raja_device_sources(Impl_Raja.cpp)

add_library(matMul_raja_lib STATIC Impl_Raja.h Impl_Raja.cpp)
target_compile_features(matMul_raja_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_raja_lib
    PUBLIC
        matMul_lib
        RAJA
)
ppb_raja_target_options(matMul_raja_lib)

# This is the main executable to be used for benchmarking purposes
add_executable(matMul_raja main.cpp)
target_compile_features(matMul_raja PUBLIC cxx_std_17)
target_link_libraries(matMul_raja
    PRIVATE
        matMul_raja_lib
)
