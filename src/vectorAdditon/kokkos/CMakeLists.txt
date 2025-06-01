add_executable(vec_kokkos Impl_Kokkos.cpp)
target_compile_features(vec_kokkos PUBLIC cxx_std_17)
target_link_libraries(vec_kokkos
    PRIVATE
        vec_lib
        Kokkos::kokkos
)