ppb_raja_device_sources(Impl_Raja.cpp)

add_executable(vec_raja Impl_Raja.cpp)
target_compile_features(vec_raja PUBLIC cxx_std_17)
target_link_libraries(vec_raja
    PRIVATE
        vec_lib
        RAJA
)
ppb_raja_target_options(vec_raja)
