add_library(matMul_acc_lib STATIC Impl_OpenACC.cpp Impl_OpenACC.h)
target_compile_features(matMul_acc_lib PUBLIC cxx_std_17)
target_link_libraries(matMul_acc_lib
        PUBLIC
        matMul_lib
        OpenACC::OpenACC_CXX
)

add_executable(matMul_acc main.cpp)
target_link_libraries(matMul_acc PUBLIC matMul_acc_lib)