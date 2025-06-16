add_library(nbody_kokkos_lib STATIC Impl_Kokkos.cpp Impl_Kokkos.h)
target_compile_features(nbody_kokkos_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_kokkos_lib
    PUBLIC
        nbody_lib
        Kokkos::kokkos
)

add_executable(nbody_kokkos main.cpp)
target_compile_features(nbody_kokkos PUBLIC cxx_std_20)
target_link_libraries(nbody_kokkos
    PUBLIC
        nbody_kokkos_lib
)