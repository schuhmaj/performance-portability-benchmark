set(VECTOR_ADDITION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VECTOR_ADDITION_FILES VectorAddition.h)

add_library(vec_lib INTERFACE ${VECTOR_ADDITION_FILES})
target_link_libraries(vec_lib INTERFACE benchmark::benchmark common_lib)
target_include_directories(vec_lib INTERFACE ${PPB_SRC_SOURCE_DIRECTORY})


if (${PPB_ENABLE_Kokkos})
    add_subdirectory(kokkos)
endif ()

if (${PPB_ENABLE_OpenCL})
    add_subdirectory(opencl)
endif()

if (${PPB_ENABLE_Boost})
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
endif ()

if(${PPB_ENABLE_Hip})
    add_subdirectory(hip)
endif ()

if (${PPB_ENABLE_Metal})
    add_subdirectory(metal)
endif ()

if (${PPB_ENABLE_Alpaka})
    add_subdirectory(alpaka)
endif ()

if (${PPB_ENABLE_Vulkan})
    add_subdirectory(vulkan)
endif ()

if (${PPB_ENABLE_Slang_Vulkan} OR ${PPB_ENABLE_Slang_Cuda})
    add_subdirectory(slang)
endif ()

add_subdirectory(cpp)
