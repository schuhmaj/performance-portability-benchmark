add_library(polyhedral_omp_lib STATIC Impl_OpenMP.cpp)
target_compile_features(polyhedral_omp_lib PUBLIC cxx_std_17)
target_link_libraries(polyhedral_omp_lib
        PUBLIC
        polyhedral_lib
        ppb_openmp_offload
)

add_executable(polyhedral_omp ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_omp PRIVATE PARADIGM="OpenMP")
target_link_libraries(polyhedral_omp PUBLIC polyhedral_omp_lib)