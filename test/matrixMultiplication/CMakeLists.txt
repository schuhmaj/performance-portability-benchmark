# Creates the Test Target
add_library(matMul_test_lib INTERFACE MatrixMultiplicationTest.h)

# Links target breakupModelTest against gtest_main (entry point, so no own main method is required)
target_link_libraries(matMul_test_lib
    INTERFACE
        GTest::gtest
        GTest::gmock
        matMul_cpp_lib
)
target_include_directories(matMul_test_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Copies the test resources into a directory next to the test executable
file(COPY "${PPB_TEST_RESOURCES_DIRECTORY}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Adds the Tests to CTest by querying the test target executable
include(GoogleTest)

add_subdirectory(cpp)

if (CMAKE_CUDA_COMPILER AND PPB_ENABLE_Cuda)
    add_subdirectory(cuda)
endif ()

if (PPB_ENABLE_OpenMP)
    add_subdirectory(omp)
endif ()

if (${PPB_ENABLE_AdaptiveCpp})
    add_subdirectory(acpp)
endif()

if (${PPB_ENABLE_Vulkan})
    add_subdirectory(vulkan)
endif ()

if (${PPB_ENABLE_Boost})
    add_subdirectory(boost)
endif()

if (${PPB_ENABLE_OpenCL})
    add_subdirectory(opencl)
endif()