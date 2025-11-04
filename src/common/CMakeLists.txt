set(COMMON_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMMON_FILES
        AlignedAllocator.h
        TypeDefinitions.h
        UtilityContainer.h
        UtilityFloatArithmetic.h
        UtilityFloatArithmetic.cpp
        FileReader.h
        FileReader.cpp
)

add_library(common_lib STATIC ${COMMON_FILES})
target_include_directories(common_lib PUBLIC ${COMMON_ROOT_DIR})
target_compile_features(common_lib PRIVATE cxx_std_17)

if (${PPB_ENABLE_OpenCL})
    add_subdirectory(opencl)
endif()

if (${PPB_ENABLE_Vulkan})
    add_subdirectory(vulkan)
endif ()
