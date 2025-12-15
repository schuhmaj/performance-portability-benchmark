set(NBODY_SIMULATION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NBODY_SIMULATION_FILES
    NBodySimulation.h
    Particle.h
    CSVFileHandler.h
)

add_library(nbody_lib INTERFACE ${NBODY_SIMULATION_FILES})
target_include_directories(nbody_lib INTERFACE ${PPB_SRC_SOURCE_DIRECTORY})
target_link_libraries(nbody_lib INTERFACE benchmark::benchmark common_lib)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_definitions(-DGIT_HASH_STRING="${GIT_HASH}")

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

if (CMAKE_CUDA_COMPILER AND PPB_ENABLE_Cuda)
    add_subdirectory(cuda)
endif ()

if (${PPB_ENABLE_Alpaka})
    add_subdirectory(alpaka)
endif ()

if (${PPB_ENABLE_OpenMP})
    add_subdirectory(openmp)
endif ()

if (${PPB_ENABLE_Vulkan})
    if (${PPB_ENABLE_NBodySimulation_naive})
        add_subdirectory(vulkan/naive)
    endif ()
    if (${PPB_ENABLE_NBodySimulation_cells})
        add_subdirectory(vulkan/cell_lists)
    endif ()
    if (${PPB_ENABLE_NBodySimulation_verlet})
        add_subdirectory(vulkan/verlet_lists)
    endif ()
endif ()

add_subdirectory(cpp)
