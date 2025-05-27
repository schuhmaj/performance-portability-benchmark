add_executable(vec_acc Impl_OpenACC.cpp)
target_include_directories(vec_acc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vec_acc
        PRIVATE
        benchmark::benchmark
        OpenACC::OpenACC_CXX
)
set_target_properties(vec_acc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)