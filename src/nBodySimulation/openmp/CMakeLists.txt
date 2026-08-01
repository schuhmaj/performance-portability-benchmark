add_library(nbody_omp_lib STATIC Impl_OpenMP.h Impl_OpenMP.cpp)
target_compile_features(nbody_omp_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_omp_lib
        PUBLIC
        nbody_lib
        ppb_openmp_offload
)

add_executable(nbody_omp main.cpp)
target_compile_features(nbody_omp PUBLIC cxx_std_20)
target_link_libraries(nbody_omp
    PRIVATE
        nbody_omp_lib
)
