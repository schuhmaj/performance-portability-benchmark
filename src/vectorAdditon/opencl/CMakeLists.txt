add_executable(vec_ocl Impl_OpenCL.cpp util/OpenCLUtility.cpp util/OpenCLUtility.h)
target_compile_features(vec_ocl PUBLIC cxx_std_17)
target_link_libraries(vec_ocl
    PUBLIC
        OpenCL::OpenCL
    PRIVATE
        vec_lib
)