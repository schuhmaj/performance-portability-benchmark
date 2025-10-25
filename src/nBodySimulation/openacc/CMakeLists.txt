add_library(nbody_acc_lib STATIC Impl_OpenACC.h Impl_OpenACC.cpp)
target_compile_features(nbody_acc_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_acc_lib
        PUBLIC
        nbody_lib
        OpenACC::OpenACC_CXX
)

add_executable(nbody_acc main.cpp)
target_compile_features(nbody_acc PUBLIC cxx_std_20)
target_link_libraries(nbody_acc
    PRIVATE
        nbody_acc_lib
)
