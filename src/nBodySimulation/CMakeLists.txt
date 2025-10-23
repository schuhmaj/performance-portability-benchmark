set(NBODY_SIMULATION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NBODY_SIMULATION_FILES
    NBodySimulation.h
    Particle.h
    CSVFileHandler.h
)

add_library(nbody_lib INTERFACE ${NBODY_SIMULATION_FILES})
target_include_directories(nbody_lib INTERFACE ${PPB_SRC_SOURCE_DIRECTORY})
target_link_libraries(nbody_lib INTERFACE benchmark::benchmark common_lib)

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
    # add_subdirectory(openmp)
endif ()

if (${PPB_ENABLE_OpenACC})
    # add_subdirectory(openacc)
endif ()

if (CMAKE_CUDA_COMPILER AND PPB_ENABLE_Cuda)
    add_subdirectory(cuda)
endif ()

add_subdirectory(cpp)
