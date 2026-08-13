# Reuse the kernels of the plain OpenCL implementation - Boost.Compute is an
# OpenCL backend, so the very same kernel sources apply. Only the host code
# differs (Boost.Compute device vectors instead of the raw OpenCL C++ bindings).
set(POLYHEDRAL_OCL_KERNEL_DIR ${POLYHEDRAL_GRAVITY_ROOT_DIR}/opencl/kernel)

compile_opencl(
        INFILE ${POLYHEDRAL_OCL_KERNEL_DIR}/opencl_init.cl
        OUTFILE opencl_init.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        VAR_NAME KERNEL_INIT
)
compile_opencl(
        INFILE ${POLYHEDRAL_OCL_KERNEL_DIR}/opencl_eval.cl
        OUTFILE opencl_eval.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        VAR_NAME KERNEL_EVAL
)
compile_opencl(
        INFILE ${POLYHEDRAL_OCL_KERNEL_DIR}/opencl_sum.cl
        OUTFILE opencl_sum.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        VAR_NAME KERNEL_SUM
)

add_library(polyhedral_boost_lib STATIC Impl_Boost.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/opencl_init.h
        ${CMAKE_CURRENT_BINARY_DIR}/opencl_eval.h
        ${CMAKE_CURRENT_BINARY_DIR}/opencl_sum.h
)
target_compile_features(polyhedral_boost_lib PUBLIC cxx_std_17)
target_include_directories(polyhedral_boost_lib PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(polyhedral_boost_lib
        PUBLIC
        polyhedral_lib
        OpenCL::OpenCL
)
target_include_directories(polyhedral_boost_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})


add_executable(polyhedral_boost ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_boost PRIVATE PARADIGM="Boost")
target_link_libraries(polyhedral_boost polyhedral_boost_lib)
