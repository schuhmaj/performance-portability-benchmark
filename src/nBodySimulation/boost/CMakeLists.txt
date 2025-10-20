compile_opencl(
        INFILE ForceKernel.cl
        OUTFILE ForceKernel.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
)
add_library(nbody_boost_lib STATIC Impl_Boost.cpp Impl_Boost.h)
target_compile_features(nbody_boost_lib PUBLIC cxx_std_20)
target_include_directories(nbody_boost_lib PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(nbody_boost_lib
        PUBLIC
        OpenCL::OpenCL
        nbody_lib
)
target_include_directories(nbody_boost_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
add_executable(nbody_boost main.cpp)
target_link_libraries(nbody_boost PUBLIC nbody_boost_lib)