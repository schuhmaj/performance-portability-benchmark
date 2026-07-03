add_executable(vec_stdpar Impl_Stdpar.cpp)
target_compile_features(vec_stdpar PUBLIC cxx_std_20)
target_link_libraries(vec_stdpar
    PUBLIC
        ppb_stdpar_offload
    PRIVATE
        vec_lib
)
