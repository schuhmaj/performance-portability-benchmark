if (${PPB_ENABLE_Kokkos})
########################################################################################
# KOKKOS IMPLEMENTATION
########################################################################################
    add_executable(vec_kokkos VectorAddition.h KokkosImpl.cpp)
    target_compile_features(vec_kokkos PUBLIC cxx_std_17)
    target_include_directories(vec_kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_kokkos
        PRIVATE
            benchmark::benchmark
            Kokkos::kokkos
    )
endif()

if (${PPB_ENABLE_Raja})
########################################################################################
# RAJA IMPLEMENTATION
########################################################################################
    add_executable(vec_raja VectorAddition.h RajaImpl.cu)
    target_compile_options(vec_raja PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_include_directories(vec_raja PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_raja
            PRIVATE
            benchmark::benchmark
            RAJA
    )
    set_target_properties(vec_raja PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if (${PPB_ENABLE_OpenCL})
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
endif()


if (${PPB_ENABLE_AdaptiveCpp})
########################################################################################
# Adaptive CPP/ OpenSYCL Implementation
########################################################################################
    add_executable(vec_acpp VectorAddition.h AdaptiveCppImpl.cpp)
    target_compile_features(vec_acpp PUBLIC cxx_std_17)
    target_include_directories(vec_acpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    add_sycl_to_target(TARGET vec_acpp SOURCES VectorAddition.h AdaptiveCppImpl.cpp)
    target_link_libraries(vec_acpp
            PRIVATE
            benchmark::benchmark
    )

    # Working on my RTX2080 setup
    # Not working on SCCS-Cluster (Complains about missing TBB? But should not be needed)
#    add_custom_target(vec_acpp_std ALL
#        COMMAND acpp -O3 --acpp-stdpar -std=c++17 -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_BINARY_DIR}/_deps/googlebench-src/include -c ${CMAKE_CURRENT_SOURCE_DIR}/AdaptiveCppStdImpl.cpp -o AdaptiveCppStdImpl.o
#        COMMAND acpp -o vec_acpp_std AdaptiveCppStdImpl.o -L${CMAKE_BINARY_DIR}/_deps/googlebench-build/src -lbenchmark
#        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/AdaptiveCppStdImpl.cpp
#    )
endif ()

########################################################################################
# Standard C++ Implementation parallel with NVHPC
########################################################################################
if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    add_executable(vec_nvhpc_std VectorAddition.h NvhpcStdImpl.cpp)
    target_include_directories(vec_nvhpc_std PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_nvhpc_std
            PRIVATE
                benchmark::benchmark
                -stdpar
    )
    target_compile_options(vec_nvhpc_std PRIVATE -stdpar=gpu)
endif()


if (${PPB_ENABLE_OpenMP})
########################################################################################
# OpenMP Target Implementation
########################################################################################
    add_executable(vec_omp VectorAddition.h OpenMpImpl.cpp)
    target_compile_features(vec_omp PUBLIC cxx_std_20)
    target_include_directories(vec_omp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_omp
            PRIVATE
                benchmark::benchmark
                OpenMP::OpenMP_CXX
    )
endif ()

if (${PPB_ENABLE_OpenACC})
########################################################################################
# OpenACC Target Implementation
########################################################################################
    add_executable(vec_acc VectorAddition.h OpenACCImpl.cpp)
    target_include_directories(vec_acc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_acc
            PRIVATE
                benchmark::benchmark
                OpenACC::OpenACC_CXX
    )
    set_target_properties(vec_acc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

if(CMAKE_CUDA_COMPILER AND PPB_ENABLE_Cuda)
########################################################################################
# Cuda Implementation
########################################################################################
    add_executable(vec_cuda VectorAddition.h CudaImpl.cu)
    target_include_directories(vec_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_cuda
        PRIVATE
            benchmark::benchmark
    )
    set_target_properties(vec_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
########################################################################################
# Cuda 2 Implementation
########################################################################################
    add_executable(vec_cuda2 VectorAddition.h Cuda2Impl.cu)
    target_include_directories(vec_cuda2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_cuda2
            PRIVATE
            benchmark::benchmark
    )
    set_target_properties(vec_cuda2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
########################################################################################
# Cuda 3 Implementation
########################################################################################
    add_executable(vec_cuda3 VectorAddition.h Cuda3Impl.cu)
    target_include_directories(vec_cuda3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(vec_cuda3
            PRIVATE
            benchmark::benchmark
    )
    set_target_properties(vec_cuda3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
########################################################################################
# Cublas Implementation
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
# Thrust Implementation
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