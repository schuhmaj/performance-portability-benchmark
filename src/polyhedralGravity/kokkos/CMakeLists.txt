add_library(polyhedral_kokkos_lib STATIC Impl_Kokkos.cpp)
target_compile_features(polyhedral_kokkos_lib PUBLIC cxx_std_20)
target_link_libraries(polyhedral_kokkos_lib
        PUBLIC
        polyhedral_lib
        Kokkos::kokkos
)

# When Kokkos targets the CUDA backend it compiles via nvcc. Allow calling
# constexpr __host__ functions (e.g. std::array::operator[]) from device code.
if (Kokkos_ENABLE_CUDA)
    target_compile_options(polyhedral_kokkos_lib PRIVATE --expt-relaxed-constexpr)
endif ()

add_executable(polyhedral_kokkos ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_kokkos PRIVATE PARADIGM="Kokkos")
target_compile_features(polyhedral_kokkos PUBLIC cxx_std_20)
target_link_libraries(polyhedral_kokkos
        PRIVATE
        polyhedral_kokkos_lib
)

