ppb_raja_device_sources(Impl_Raja.cpp)

add_library(nbody_raja_lib STATIC Impl_Raja.cpp Impl_Raja.h)
target_compile_features(nbody_raja_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_raja_lib
    PUBLIC
        nbody_lib
        RAJA
)
ppb_raja_target_options(nbody_raja_lib)

add_executable(nbody_raja main.cpp)
target_compile_features(nbody_raja PUBLIC cxx_std_20)
target_link_libraries(nbody_raja
    PRIVATE
        nbody_raja_lib
)
