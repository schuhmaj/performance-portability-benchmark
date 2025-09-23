set(MATRIX_MULTIPLICATION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MATRIX_MULTIPLICATION_FILES MatrixMultiplication.h)

add_library(matMul_lib INTERFACE ${MATRIX_MULTIPLICATION_FILES})
target_link_libraries(matMul_lib INTERFACE common_lib benchmark::benchmark )
target_include_directories(matMul_lib INTERFACE ${MATRIX_MULTIPLICATION_ROOT_DIR})

if (CMAKE_CUDA_COMPILER AND PPB_ENABLE_Cuda)
    add_subdirectory(cuda)
endif ()

if (${PPB_ENABLE_OpenMP})
    add_subdirectory(openmp)
endif ()

if (${PPB_ENABLE_Kokkos})
    add_subdirectory(kokkos)
endif ()

if (${PPB_ENABLE_Boost})
    add_subdirectory(boost)
endif()

if (${PPB_ENABLE_OpenCL})
    add_subdirectory(opencl)
endif()

add_subdirectory(cpp)
