add_executable(vec_kokkos Impl_Kokkos.cpp)
target_compile_features(vec_kokkos PUBLIC cxx_std_17)
target_include_directories(vec_kokkos PUBLIC ${VECTOR_ADDITION_ROOT_DIR})
target_link_libraries(vec_kokkos
        PRIVATE
        benchmark::benchmark
        Kokkos::kokkos
)