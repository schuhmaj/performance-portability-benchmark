# Copies the test resources into a directory next to the test executable
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../resources" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

function(create_polyhedral_test NAME)
    add_executable(polyhedral_${NAME}_test GravityModelTest.cpp TestMain.cpp)

    target_link_libraries(polyhedral_${NAME}_test
            GTest::gtest
            GTest::gmock
            polyhedral_${NAME}_lib
    )
    gtest_discover_tests(polyhedral_${NAME}_test DISCOVERY_MODE POST_BUILD)
endfunction()


create_polyhedral_test(cpp)

if (${PPB_ENABLE_CUDA})
    create_polyhedral_test(cuda)
endif ()

if (${PPB_ENABLE_HIP})
    create_polyhedral_test(hip)
endif ()

if (PPB_ENABLE_OpenMP)
    create_polyhedral_test(omp)
endif ()

if (PPB_ENABLE_Stdpar)
    create_polyhedral_test(stdpar)
endif ()

if (${PPB_ENABLE_AdaptiveCpp})
    create_polyhedral_test(acpp)
endif()

if (${PPB_ENABLE_Vulkan})
    create_polyhedral_test(vulkan)
endif ()

if (${PPB_ENABLE_Boost})
    create_polyhedral_test(boost)
endif()

if (${PPB_ENABLE_OpenCL})
    create_polyhedral_test(ocl)
endif()

if (${PPB_ENABLE_Alpaka})
    create_polyhedral_test(alpaka)
endif ()

if (${PPB_ENABLE_Kokkos})
    create_polyhedral_test(kokkos)
endif ()

if (${PPB_ENABLE_OpenACC})
    create_polyhedral_test(acc)
endif ()

if (${PPB_ENABLE_Slang_Vulkan})
    create_polyhedral_test(slang_vulkan)
endif ()

if (${PPB_ENABLE_Slang_Cuda})
    create_polyhedral_test(slang_cuda)
endif ()