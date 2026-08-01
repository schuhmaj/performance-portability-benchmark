set(COMMON_NBODY_TEST_SRC
    GoogleTestMatcher.h
    NBodyTest.h
)
# Creates the Test Target
add_library(nbody_test_lib INTERFACE ${COMMON_NBODY_TEST_SRC})

# Links target breakupModelTest against gtest_main (entry point, so no own main method is required)
target_link_libraries(nbody_test_lib
    INTERFACE
        GTest::gtest
        GTest::gmock
        nbody_cpp_lib
)
target_include_directories(nbody_test_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Copies the test resources into a directory next to the test executable
file(COPY "${PPB_TEST_RESOURCES_DIRECTORY}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Adds the Tests to CTest by querying the test target executable
include(GoogleTest)

add_subdirectory(cpp)

if (${PPB_ENABLE_Kokkos})
    add_subdirectory(kokkos)
endif ()

if (${PPB_ENABLE_Boost})
    add_subdirectory(boost)
endif()

if (${PPB_ENABLE_OpenCL})
    add_subdirectory(opencl)
endif()

if (CMAKE_CUDA_COMPILER AND PPB_ENABLE_CUDA)
    add_subdirectory(cuda)
    add_compile_options(-gpu=lineinfo)
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

if (${PPB_ENABLE_Slang_Vulkan})
    if (${PPB_ENABLE_NBodySimulation_naive})
        add_subdirectory(slang/vulkan/naive)
    endif ()
    if (${PPB_ENABLE_NBodySimulation_cells})
        add_subdirectory(slang/vulkan/cell_lists)
    endif ()
    if (${PPB_ENABLE_NBodySimulation_verlet})
        add_subdirectory(slang/vulkan/verlet_lists)
    endif ()
endif ()

if (${PPB_ENABLE_Slang_Cuda})
    if (${PPB_ENABLE_NBodySimulation_naive})
        add_subdirectory(slang/cuda/naive)
    endif ()
    if (${PPB_ENABLE_NBodySimulation_cells})
        add_subdirectory(slang/cuda/cell_lists)
    endif ()
    if (${PPB_ENABLE_NBodySimulation_verlet})
        add_subdirectory(slang/cuda/verlet_lists)
    endif ()
endif ()

if (${PPB_ENABLE_AdaptiveCpp})
    add_subdirectory(acpp)
endif()

if (${PPB_ENABLE_OpenACC})
    add_subdirectory(openacc)
endif ()

if (${PPB_ENABLE_HIP})
    add_subdirectory(hip)
endif ()