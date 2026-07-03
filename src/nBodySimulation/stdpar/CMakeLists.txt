# The Stdpar library can be linked to e.g. a test executable
add_library(nbody_stdpar_lib STATIC Impl_Stdpar.h Impl_Stdpar.cpp)
target_compile_features(nbody_stdpar_lib PUBLIC cxx_std_20)
target_link_libraries(nbody_stdpar_lib
    PUBLIC
        nbody_lib
        ppb_stdpar_offload
)

# This is the main executable to be used for benchmarking purposes
add_executable(nbody_stdpar main.cpp)
target_compile_features(nbody_stdpar PUBLIC cxx_std_20)
target_link_libraries(nbody_stdpar
    PRIVATE
        nbody_stdpar_lib
)
