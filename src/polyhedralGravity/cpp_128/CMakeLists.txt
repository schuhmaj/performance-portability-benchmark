find_package(OpenMP REQUIRED)
add_exe(cpp_128 FORCE_BITS 128 LIBS OpenMP::OpenMP_CXX)
