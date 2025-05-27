add_executable(vec_omp impl_openmp.cpp)
target_compile_features(vec_omp PUBLIC cxx_std_20)
target_include_directories(vec_omp PUBLIC ${VECTOR_ADDITION_ROOT_DIR})
target_link_libraries(vec_omp
        PRIVATE
        benchmark::benchmark
        OpenMP::OpenMP_CXX
)
target_compile_options(vec_omp PRIVATE -mp=gpu)