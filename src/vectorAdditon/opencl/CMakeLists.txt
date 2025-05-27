add_executable(vec_ocl impl_opencl.cpp util/OpenCLUtility.cpp util/OpenCLUtlity.h)
target_compile_features(vec_ocl PUBLIC cxx_std_20)
target_include_directories(vec_ocl PUBLIC ${VECTOR_ADDITION_ROOT_DIR})
target_link_libraries(vec_ocl
        PUBLIC
        OpenCL::OpenCL
        PRIVATE
        benchmark::benchmark
)