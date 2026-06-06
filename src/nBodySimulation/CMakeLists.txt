set(NBODY_SIMULATION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NBODY_SIMULATION_FILES
    NBodySimulation.h
    Particle.h
    CSVFileHandler.h
)

add_library(nbody_lib INTERFACE ${NBODY_SIMULATION_FILES})
target_include_directories(nbody_lib INTERFACE ${PPB_SRC_SOURCE_DIRECTORY})
target_link_libraries(nbody_lib INTERFACE benchmark::benchmark common_lib)

if (${PPB_ENABLE_VTK})
# taken from https://github.com/MaxieHelenBichmann/MolSim-WS25-GroupD/blob/main/cmake/modules/vtk.cmake
    find_package(VTK REQUIRED COMPONENTS
            CommonCore
            CommonDataModel
            IOXML
    )

    if(VTK_FOUND)
        message (STATUS "Found VTK Version: ${VTK_VERSION}")
    else ()
        message(FATAL_ERROR "VTK not found")
    endif ()

    if(VTK_VERSION VERSION_GREATER_EQUAL 8.9)
        target_include_directories(nbody_lib INTERFACE ${VTK_INCLUDE_DIRS})
    else()
        include(${VTK_USE_FILE})
    endif ()

    target_link_libraries(nbody_lib
            INTERFACE
            ${VTK_LIBRARIES}
    )
    add_compile_definitions(PPB_ENABLE_VTK)
endif ()

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
    add_subdirectory(vulkan)
endif ()

if (${PPB_ENABLE_OpenACC})
    add_subdirectory(openacc)
endif ()

add_subdirectory(cpp)
