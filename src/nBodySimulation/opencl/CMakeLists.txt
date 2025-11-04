compile_opencl(
        INFILE ForceKernel.cl
        OUTFILE ForceKernel.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
)
add_library(nbody_ocl_lib STATIC Impl_OpenCL.cpp Impl_OpenCL.h "${CMAKE_CURRENT_BINARY_DIR}/ForceKernel.h")
target_compile_features(nbody_ocl_lib PUBLIC cxx_std_20)
target_include_directories(nbody_ocl_lib PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(nbody_ocl_lib
        PUBLIC
        common_ocl_lib
        nbody_lib
)
target_include_directories(nbody_ocl_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
add_executable(nbody_ocl main.cpp)
target_link_libraries(nbody_ocl PUBLIC nbody_ocl_lib)