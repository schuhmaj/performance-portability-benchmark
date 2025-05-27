add_executable(vec_boost Impl_Boost.cpp)
target_compile_features(vec_boost PUBLIC cxx_std_14)
target_include_directories(vec_boost PUBLIC ${VECTOR_ADDITION_ROOT_DIR} ${Boost_INCLUDE_DIR})
target_link_libraries(vec_boost
        PUBLIC
        OpenCL::OpenCL
        PRIVATE
        benchmark::benchmark
)