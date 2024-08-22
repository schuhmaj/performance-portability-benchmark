########################################################################################
# KOKKOS IMPLEMENTATION
########################################################################################
add_executable(vec_kokkos VectorAddition.h KokkosImpl.cpp)
target_compile_features(vec_kokkos PUBLIC cxx_std_20)
target_include_directories(vec_kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vec_kokkos
    PRIVATE
        benchmark::benchmark
        Kokkos::kokkos
)
########################################################################################
# Boost Compute Implementation
########################################################################################
add_executable(vec_boost VectorAddition.h BoostComputeImpl.cpp)
target_compile_features(vec_boost PUBLIC cxx_std_14)
target_include_directories(vec_boost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIR})
target_link_libraries(vec_boost
    PUBLIC
        OpenCL::OpenCL
    PRIVATE
        benchmark::benchmark
)