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