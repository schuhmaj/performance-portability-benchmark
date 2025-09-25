add_executable(matMul_boost Impl_Boost.cpp Impl_Boost.h main.cpp)
target_compile_features(matMul_boost PUBLIC cxx_std_14)
target_include_directories(matMul_boost PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(matMul_boost
        PUBLIC
        OpenCL::OpenCL
        PRIVATE
        vec_lib
)