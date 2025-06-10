add_executable(nbody_cpp Impl_Cpp.cpp)
target_compile_features(nbody_cpp PUBLIC cxx_std_17)
target_link_libraries(nbody_cpp
        PRIVATE
        nbody_lib
)