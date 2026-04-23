add_library(polyhedral_acc_lib STATIC Impl_OpenACC.cpp)
target_compile_features(polyhedral_acc_lib PUBLIC cxx_std_17)
target_link_libraries(polyhedral_acc_lib
        PUBLIC
        polyhedral_lib
        OpenACC::OpenACC_CXX
)

add_executable(polyhedral_acc ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_link_libraries(polyhedral_acc PUBLIC polyhedral_acc_lib)
