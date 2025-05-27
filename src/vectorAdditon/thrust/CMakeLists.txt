add_executable(vec_thrust Impl_Thrust.cu)
target_include_directories(vec_thrust PUBLIC ${VECTOR_ADDITION_ROOT_DIR})
target_link_libraries(vec_thrust
        PRIVATE
        Thrust
        benchmark::benchmark
)
set_target_properties(vec_thrust PROPERTIES CUDA_SEPARABLE_COMPILATION ON)