add_library(matMul_omp_lib STATIC Impl_OpenMPDevice.cpp Impl_OpenMPDevice.h Impl_OpenMP.cpp Impl_OpenMP.h)
target_compile_features(matMul_omp_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_omp_lib
        PUBLIC
        matMul_lib
        OpenMP::OpenMP_CXX
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH omp_lib_dir)
    target_compile_options(matMul_omp_lib PUBLIC -fopenmp-targets=nvptx64-nvidia-cuda)
    target_link_options(matMul_omp_lib PUBLIC -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda "-Wl,-rpath,${omp_lib_dir}")
endif ()



add_executable(matMul_omp main.cpp)
target_link_libraries(matMul_omp PUBLIC matMul_omp_lib)