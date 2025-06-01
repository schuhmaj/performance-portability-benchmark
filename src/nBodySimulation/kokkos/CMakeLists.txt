add_executable(nbody_kokkos Impl_Kokkos.cpp)
target_compile_features(nbody_kokkos PUBLIC cxx_std_17)
target_include_directories(nbody_kokkos PUBLIC ${NBODY_SIMULATION_ROOT_DIR})
target_link_libraries(nbody_kokkos
        PRIVATE
        benchmark::benchmark
        Kokkos::kokkos
)