add_library(matMul_omp_lib STATIC Impl_OpenMPDevice.cpp Impl_OpenMPDevice.h Impl_OpenMP.cpp Impl_OpenMP.h)
target_compile_features(matMul_omp_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_omp_lib
        PUBLIC
        matMul_lib
        ppb_openmp_offload
)

add_executable(matMul_omp main.cpp)
target_link_libraries(matMul_omp PUBLIC matMul_omp_lib)