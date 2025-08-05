add_library(matMul_kokkos_lib STATIC Impl_Kokkos.h Impl_Kokkos.cpp)
target_compile_features(matMul_kokkos_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_kokkos_lib
    PUBLIC
        matMul_lib
        Kokkos::kokkos
)

# This is the main executable to be used for benchmarking purposes
add_executable(matMul_kokkos main.cpp)
target_compile_features(matMul_kokkos PUBLIC cxx_std_17)
target_link_libraries(matMul_kokkos
    PRIVATE
        matMul_kokkos_lib
)
