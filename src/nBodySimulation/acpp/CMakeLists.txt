# The CPP library can be linked to e.g. a test executable
add_library(nbody_acpp_lib STATIC Impl_AdaptiveCpp.h Impl_AdaptiveCpp.cpp)
target_compile_features(nbody_acpp_lib PUBLIC cxx_std_17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(nbody_acpp PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    target_link_options(nbody_acpp PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
else ()
    add_sycl_to_target(TARGET nbody_acpp_lib)
endif ()
target_link_libraries(nbody_acpp_lib
    PUBLIC
        nbody_lib
)

# This is the main executable to be used for benchmarking purposes
add_executable(nbody_acpp main.cpp)
target_compile_features(nbody_acpp PUBLIC cxx_std_17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(nbody_acpp PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
    target_link_options(nbody_acpp PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
else ()
    add_sycl_to_target(TARGET nbody_acpp)
endif ()
target_link_libraries(nbody_acpp
    PRIVATE
        nbody_acpp_lib
)
