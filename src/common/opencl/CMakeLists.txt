set(COMMON_OCL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_library(common_ocl_lib STATIC OpenCLUtility.h OpenCLUtility.cpp)
target_link_libraries(common_ocl_lib PUBLIC OpenCL::OpenCL)
target_include_directories(common_ocl_lib PUBLIC ${COMMON_OCL_ROOT_DIR})
