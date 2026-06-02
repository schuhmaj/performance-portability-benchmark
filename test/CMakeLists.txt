include(gtest)

set(PPB_TEST_RESOURCES_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources)

# Convenience script that runs every GoogleTest binary directly (much faster to
# start than `ctest`, which re-discovers tests via PRE_TEST on each invocation).
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/run_all_tests.sh.in
        ${CMAKE_BINARY_DIR}/run_all_tests.sh
        @ONLY)
file(CHMOD ${CMAKE_BINARY_DIR}/run_all_tests.sh
        PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

if (${PPB_ENABLE_NBodySimulation})
    add_subdirectory(nBodySimulation)
endif ()

if (${PPB_ENABLE_MatrixMultiplication})
    add_subdirectory(matrixMultiplication)
endif ()

if (${PPB_ENABLE_PolyhedralGravity})
    add_subdirectory(polyhedralGravity)
endif ()