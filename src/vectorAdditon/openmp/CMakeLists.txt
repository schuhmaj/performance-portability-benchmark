add_executable(vec_omp Impl_OpenMP.cpp)
target_compile_features(vec_omp PUBLIC cxx_std_20)
target_link_libraries(vec_omp
    PUBLIC
        OpenMP::OpenMP_CXX
    PRIVATE
        vec_lib

)
cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH omp_lib_dir)
target_compile_options(vec_omp PUBLIC -fopenmp-targets=nvptx64-nvidia-cuda)
target_link_options(vec_omp PUBLIC -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda "-Wl,-rpath,${omp_lib_dir}")