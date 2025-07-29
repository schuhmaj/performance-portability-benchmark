include(gtest)

set(PPB_TEST_RESOURCES_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources)

if (${PPB_ENABLE_NBodySimulation})
    add_subdirectory(nBodySimulation)
endif ()

if (${PPB_ENABLE_MatrixMultiplication})
    add_subdirectory(matrixMultiplication)
endif ()