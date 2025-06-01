add_executable(vec_ocl impl_opencl.cpp util/OpenCLUtility.cpp util/OpenCLUtlity.h)
target_compile_features(vec_ocl PUBLIC cxx_std_20)
target_link_libraries(vec_ocl
    PUBLIC
        OpenCL::OpenCL
    PRIVATE
        vec_lib
)