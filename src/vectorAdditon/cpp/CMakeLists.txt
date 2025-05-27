add_executable(vec_cstd Impl_StandardCPP.cpp)
target_include_directories(vec_cstd PUBLIC ${VECTOR_ADDITION_ROOT_DIR})
target_compile_features(vec_cstd PUBLIC cxx_std_17)
target_link_libraries(vec_cstd
        PRIVATE
        benchmark::benchmark
)