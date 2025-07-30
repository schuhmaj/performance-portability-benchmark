# This is the main executable to be used for benchmarking purposes
add_executable(matMul_omp main.cpp Impl_OpenMP.cpp Impl_OpenMP.h)
target_compile_features(matMul_omp PUBLIC cxx_std_20)
target_link_libraries(matMul_omp
        PRIVATE
        matMul_lib
        OpenMP::OpenMP_CXX
)

add_executable(matMul_omp_dev mainDevice.cpp Impl_OpenMPDevice.cpp Impl_OpenMPDevice.h)
target_compile_features(matMul_omp_dev PUBLIC cxx_std_20)
target_link_libraries(matMul_omp_dev
        PRIVATE
        matMul_lib
        OpenMP::OpenMP_CXX
)
target_compile_options(matMul_omp_dev PUBLIC -fopenmp-targets=nvptx64-nvidia-cuda)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH omp_lib_dir)
    target_link_options(matMul_omp_dev PUBLIC -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda "-Wl,-rpath,${omp_lib_dir}")
endif ()
