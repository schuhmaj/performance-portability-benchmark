add_executable(vec_omp Impl_OpenMP.cpp)
target_compile_features(vec_omp PUBLIC cxx_std_20)
target_link_libraries(vec_omp
    PUBLIC
        ppb_openmp_offload
    PRIVATE
        vec_lib

)