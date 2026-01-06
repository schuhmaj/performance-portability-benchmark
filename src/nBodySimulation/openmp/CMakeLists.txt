add_library(nbody_omp_lib STATIC Impl_OpenMP.h Impl_OpenMP.cpp)
target_compile_features(nbody_omp_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_omp_lib
        PUBLIC
        nbody_lib
        OpenMP::OpenMP_CXX
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH omp_lib_dir)
    target_compile_options(nbody_omp_lib PUBLIC -fopenmp-targets=nvptx64 -fopenmp-target-jit)
    target_link_options(nbody_omp_lib PUBLIC -fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-jit "-Wl,-rpath,${omp_lib_dir}")
endif ()

add_executable(nbody_omp main.cpp)
target_compile_features(nbody_omp PUBLIC cxx_std_20)
target_link_libraries(nbody_omp
    PRIVATE
        nbody_omp_lib
)
