add_executable(vec_thrust Impl_Thrust.cu)
target_link_libraries(vec_thrust
    PRIVATE
        Thrust
        vec_lib
)
set_target_properties(vec_thrust PROPERTIES CUDA_SEPARABLE_COMPILATION ON)