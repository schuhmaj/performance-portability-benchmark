set(MATRIX_MULTIPLICATION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MATRIX_MULTIPLICATION_FILES MatrixMultiplication.h)

add_library(matMul_lib INTERFACE ${MATRIX_MULTIPLICATION_FILES})
target_link_libraries(matMul_lib INTERFACE common_lib benchmark::benchmark )
target_include_directories(matMul_lib INTERFACE ${PPB_SRC_SOURCE_DIRECTORY})

if (CMAKE_CUDA_COMPILER AND PPB_ENABLE_CUDA)
    add_subdirectory(cuda)
endif ()

if (${PPB_ENABLE_OpenMP})
    add_subdirectory(openmp)
endif ()

if (${PPB_ENABLE_Stdpar})
    add_subdirectory(stdpar)
endif ()

if (${PPB_ENABLE_Kokkos})
    add_subdirectory(kokkos)
endif ()

if (${PPB_ENABLE_Raja})
    add_subdirectory(raja)
endif ()

if (${PPB_ENABLE_Boost})
    add_subdirectory(boost)
endif()

if (${PPB_ENABLE_OpenCL})
    add_subdirectory(opencl)
endif()

if (${PPB_ENABLE_AdaptiveCpp})
    add_subdirectory(acpp)
endif()

if (${PPB_ENABLE_Alpaka})
    add_subdirectory(alpaka)
endif ()

if (${PPB_ENABLE_Vulkan})
    add_subdirectory(vulkan)
endif ()

if (${PPB_ENABLE_Slang_Vulkan} OR ${PPB_ENABLE_Slang_Cuda})
    add_subdirectory(slang)
endif ()

if (${PPB_ENABLE_OpenACC})
    add_subdirectory(openacc)
endif ()

if (${PPB_ENABLE_HIP})
    add_subdirectory(hip)
endif ()

add_subdirectory(cpp)
