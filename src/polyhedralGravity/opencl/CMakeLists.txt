compile_opencl(
        INFILE ${CMAKE_CURRENT_SOURCE_DIR}/kernel/opencl_init.cl
        OUTFILE opencl_init.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        VAR_NAME KERNEL_INIT
)
compile_opencl(
        INFILE ${CMAKE_CURRENT_SOURCE_DIR}/kernel/opencl_eval.cl
        OUTFILE opencl_eval.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        VAR_NAME KERNEL_EVAL
)
compile_opencl(
        INFILE ${CMAKE_CURRENT_SOURCE_DIR}/kernel/opencl_sum.cl
        OUTFILE opencl_sum.h
        PATH ${CMAKE_CURRENT_BINARY_DIR}
        VAR_NAME KERNEL_SUM
)

add_library(polyhedral_ocl_lib STATIC Impl_OpenCL.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/opencl_init.h
        ${CMAKE_CURRENT_BINARY_DIR}/opencl_eval.h
        ${CMAKE_CURRENT_BINARY_DIR}/opencl_sum.h
)
target_compile_features(polyhedral_ocl_lib PUBLIC cxx_std_17)
target_link_libraries(polyhedral_ocl_lib
        PUBLIC
        polyhedral_lib
        common_ocl_lib
)
target_include_directories(polyhedral_ocl_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})


add_executable(polyhedral_ocl ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_ocl PRIVATE PARADIGM="OpenCL")
target_link_libraries(polyhedral_ocl polyhedral_ocl_lib)
