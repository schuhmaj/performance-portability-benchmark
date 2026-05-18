add_library(polyhedral_kokkos_lib STATIC Impl_Kokkos.cpp)
target_compile_features(polyhedral_kokkos_lib PUBLIC cxx_std_20)
target_link_libraries(polyhedral_kokkos_lib
        PUBLIC
        polyhedral_lib
        Kokkos::kokkos
)

add_executable(polyhedral_kokkos ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_kokkos PRIVATE PARADIGM="Kokkos")
target_compile_features(polyhedral_kokkos PUBLIC cxx_std_20)
target_link_libraries(polyhedral_kokkos
        PRIVATE
        polyhedral_kokkos_lib
)

