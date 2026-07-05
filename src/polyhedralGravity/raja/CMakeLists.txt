ppb_raja_device_sources(Impl_Raja.cpp)

add_library(polyhedral_raja_lib STATIC Impl_Raja.cpp)
target_compile_features(polyhedral_raja_lib PUBLIC cxx_std_20)
target_link_libraries(polyhedral_raja_lib
        PUBLIC
        polyhedral_lib
        RAJA
)
ppb_raja_target_options(polyhedral_raja_lib)

add_executable(polyhedral_raja ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_raja PRIVATE PARADIGM="RAJA")
target_compile_features(polyhedral_raja PUBLIC cxx_std_20)
target_link_libraries(polyhedral_raja
        PRIVATE
        polyhedral_raja_lib
)
