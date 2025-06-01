add_executable(vec_omp impl_openmp.cpp)
target_compile_features(vec_omp PUBLIC cxx_std_20)
target_link_libraries(vec_omp
    PUBLIC
        OpenMP::OpenMP_CXX
    PRIVATE
        vec_lib

)
target_compile_options(vec_omp PRIVATE -mp=gpu)