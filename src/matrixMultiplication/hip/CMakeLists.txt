add_library(matMul_hip_lib STATIC Impl_HipNaive.hip Impl_HipNaive.h)
target_link_libraries(matMul_hip_lib PUBLIC matMul_lib)

add_executable(matMul_hip main.hip)
target_link_libraries(matMul_hip PRIVATE matMul_hip_lib)
