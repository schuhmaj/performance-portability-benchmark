add_executable(vec_acc Impl_OpenACC.cpp)
target_link_libraries(vec_acc
        PRIVATE
        vec_lib
        OpenACC::OpenACC_CXX
)