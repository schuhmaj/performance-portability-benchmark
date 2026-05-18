add_library(polyhedral_omp_lib STATIC Impl_OpenMP.cpp)
target_compile_features(polyhedral_omp_lib PUBLIC cxx_std_17)
target_link_libraries(polyhedral_omp_lib
        PUBLIC
        polyhedral_lib
        OpenMP::OpenMP_CXX
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH omp_lib_dir)
    target_compile_options(polyhedral_omp_lib PUBLIC -fopenmp-targets=nvptx64 -fopenmp-target-jit)
    target_link_options(polyhedral_omp_lib PUBLIC -fopenmp -fopenmp-targets=nvptx64 -fopenmp-target-jit "-Wl,-rpath,${omp_lib_dir}")
endif ()

add_executable(polyhedral_omp ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_omp PRIVATE PARADIGM="OpenMP")
target_link_libraries(polyhedral_omp PUBLIC polyhedral_omp_lib)