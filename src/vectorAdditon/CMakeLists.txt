set(VECTOR_ADDITION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VECTOR_ADDITION_FILES VectorAddition.h)

add_library(vec_lib INTERFACE ${VECTOR_ADDITION_FILES})
target_link_libraries(vec_lib INTERFACE benchmark::benchmark)
target_include_directories(vec_lib INTERFACE ${VECTOR_ADDITION_ROOT_DIR})


if (${PPB_ENABLE_Kokkos})
    add_subdirectory(kokkos)
endif ()

if (${PPB_ENABLE_OpenCL})
    add_subdirectory(opencl)
    add_subdirectory(boost)
endif()

if (${PPB_ENABLE_AdaptiveCpp})
    add_subdirectory(acpp)
endif()

if (${PPB_ENABLE_OpenMP})
    add_subdirectory(openmp)
endif ()

if (${PPB_ENABLE_OpenACC})
    add_subdirectory(openacc)
endif ()

if(CMAKE_CUDA_COMPILER AND PPB_ENABLE_Cuda)
    add_subdirectory(cuda)
    add_subdirectory(thrust)
endif ()

if (${PPB_ENABLE_MetalCpp})
    add_subdirectory(metal)
endif ()

add_subdirectory(cpp)
