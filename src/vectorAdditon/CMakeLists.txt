########################################################################################
# KOKKOS IMPLEMENTATION
########################################################################################
add_executable(vec_kokkos VectorAddition.h KokkosImpl.cpp)
target_compile_features(vec_kokkos PUBLIC cxx_std_20)
target_include_directories(vec_kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vec_kokkos
    PRIVATE
        benchmark::benchmark
        Kokkos::kokkos
)
########################################################################################
# Boost Compute Implementation
########################################################################################
add_executable(vec_boost VectorAddition.h BoostComputeImpl.cpp)
target_compile_features(vec_boost PUBLIC cxx_std_14)
target_include_directories(vec_boost PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIR})
target_link_libraries(vec_boost
    PUBLIC
        OpenCL::OpenCL
    PRIVATE
        benchmark::benchmark
)
########################################################################################
# Raw OpenCL Implementation
########################################################################################
add_executable(vec_ocl VectorAddition.h OpenCLImp.cpp util/OpenCLUtility.cpp util/OpenCLUtlity.h)
target_compile_features(vec_ocl PUBLIC cxx_std_20)
target_include_directories(vec_ocl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vec_ocl
        PUBLIC
            OpenCL::OpenCL
        PRIVATE
            benchmark::benchmark
)
########################################################################################
# Standard C++ Implementation
########################################################################################
add_executable(vec_std VectorAddition.h CStdImpl.cpp)
target_compile_features(vec_std PUBLIC cxx_std_20)
target_include_directories(vec_std PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vec_std
        PRIVATE
            benchmark::benchmark
)
if(CMAKE_CUDA_COMPILER)
########################################################################################
# Raw Cuda Implementation
########################################################################################
    add_executable(vec_cuda VectorAddition.h CudaImpl.cu)
    target_include_directories(vec_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_cuda
        PRIVATE
            benchmark::benchmark
    )
    set_target_properties(vec_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
########################################################################################
# Raw Cuda Implementation
########################################################################################
add_executable(vec_cublas VectorAddition.h CublasImpl.cu)
target_include_directories(vec_cublas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vec_cublas
        PRIVATE
        benchmark::benchmark
        CUDA::cublas
)
set_target_properties(vec_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
########################################################################################
# Thrust Cuda Implementation
########################################################################################
    add_executable(vec_thrust VectorAddition.h ThrustImpl.cu)
    target_include_directories(vec_thrust PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_thrust
        PRIVATE
            Thrust
            benchmark::benchmark
    )
    set_target_properties(vec_thrust PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()