add_executable(vec_kokkos VectorAddition.h KokkosImpl.cpp)
target_include_directories(vec_kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vec_kokkos
    PRIVATE
        benchmark::benchmark
        Kokkos::kokkos
)