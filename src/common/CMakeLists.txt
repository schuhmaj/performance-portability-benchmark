set(COMMON_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMMON_FILES
        AlignedAllocator.h
        ContainerUtility.h
        TypeDefinitions.h
        UtilityContainer.h
        UtilityFloatArithmetic.h
        UtilityFloatArithmetic.cpp)

add_library(common_lib STATIC ${COMMON_FILES})
target_include_directories(common_lib PUBLIC ${COMMON_ROOT_DIR})
