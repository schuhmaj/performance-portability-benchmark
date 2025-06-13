add_executable(nbody_cpp Impl_Cpp.cpp benchmark_main.cpp)
target_compile_features(nbody_cpp PUBLIC cxx_std_20)
target_link_libraries(nbody_cpp
        PRIVATE
        nbody_lib
)

add_executable(ref_nbody Impl_Cpp.cpp output_main.cpp)
target_compile_features(ref_nbody PUBLIC cxx_std_20)
target_link_libraries(ref_nbody
        PRIVATE
        nbody_lib
)