# The Stdpar library can be linked to e.g. a test executable
add_library(polyhedral_stdpar_lib STATIC Impl_Stdpar.cpp)
target_compile_features(polyhedral_stdpar_lib PUBLIC cxx_std_20)
target_link_libraries(polyhedral_stdpar_lib
        PUBLIC
        polyhedral_lib
        ppb_stdpar_offload
)

# This is the main executable to be used for benchmarking purposes
add_executable(polyhedral_stdpar ${POLYHEDRAL_GRAVITY_ROOT_DIR}/main.cpp)
target_compile_definitions(polyhedral_stdpar PRIVATE PARADIGM="Stdpar")
target_compile_features(polyhedral_stdpar PUBLIC cxx_std_20)
target_link_libraries(polyhedral_stdpar
        PRIVATE
        polyhedral_stdpar_lib
)
