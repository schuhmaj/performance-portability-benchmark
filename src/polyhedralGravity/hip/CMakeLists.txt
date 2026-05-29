add_library(polyhedral_hip_lib STATIC Impl_Hip.hip)
target_link_libraries(polyhedral_hip_lib PUBLIC polyhedral_lib)
target_compile_features(polyhedral_hip_lib PUBLIC cxx_std_17)
set_target_properties(polyhedral_hip_lib PROPERTIES
        HIP_SEPARABLE_COMPILATION ON
)
target_compile_options(polyhedral_hip_lib PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-arch=native>)


add_executable(polyhedral_hip ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_hip PRIVATE PARADIGM="Hip")
target_link_libraries(polyhedral_hip PRIVATE polyhedral_hip_lib)
target_compile_features(polyhedral_hip PUBLIC cxx_std_17)
