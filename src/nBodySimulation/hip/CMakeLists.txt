add_library(nbody_hip_lib STATIC Impl_Hip.hip Impl_Hip.h)
target_compile_features(nbody_hip_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_hip_lib PUBLIC nbody_lib)
set_target_properties(nbody_hip_lib PROPERTIES
        HIP_ARCHITECTURES native
)

add_executable(nbody_hip main.hip)
target_compile_features(nbody_hip PUBLIC cxx_std_20)
target_link_libraries(nbody_hip PRIVATE nbody_hip_lib)
set_target_properties(nbody_hip PROPERTIES
        HIP_ARCHITECTURES native
)
