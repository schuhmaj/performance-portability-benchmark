add_executable(nbody_kokkos Impl_Kokkos.cpp)
target_compile_features(nbody_kokkos PUBLIC cxx_std_17)
target_link_libraries(nbody_kokkos
    PRIVATE
        nbody_lib
        Kokkos::kokkos
)