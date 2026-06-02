set(POLYHEDRAL_GRAVITY_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLYHEDRAL_GRAVITY_FILES PolyhedralGravityDefinitions.h)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
include(tetgen)

add_library(polyhedral_lib INTERFACE ${POLYHEDRAL_GRAVITY_FILES})
target_link_libraries(polyhedral_lib INTERFACE benchmark::benchmark tetgen::tetgen common_lib)
target_include_directories(polyhedral_lib INTERFACE ${PPB_SRC_SOURCE_DIRECTORY})
target_compile_definitions(polyhedral_lib INTERFACE FLOAT_BITS=${PPB_FloatType})

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

if (${PPB_ENABLE_Slang_Vulkan} OR ${PPB_ENABLE_Slang_Cuda})
    add_subdirectory(slang)
endif()

if (${PPB_ENABLE_OpenMP})
    add_subdirectory(openmp)
endif ()

if (${PPB_ENABLE_OpenACC})
    add_subdirectory(openacc)
endif ()

if(CMAKE_CUDA_COMPILER AND PPB_ENABLE_CUDA)
    add_subdirectory(cuda)
endif ()

if (${PPB_ENABLE_HIP})
    add_subdirectory(hip)
endif ()

if (${PPB_ENABLE_Alpaka})
    add_subdirectory(alpaka)
endif ()

if (${PPB_ENABLE_Vulkan})
    add_subdirectory(vulkan)
endif ()

add_subdirectory(cpp)
