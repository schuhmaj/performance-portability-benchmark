cmake_minimum_required(VERSION 3.22)
project(performance_portability_benchmark VERSION 1.0 LANGUAGES CXX)

include(CheckLanguage)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA language enabled and CUDAToolkit found")
    else ()
        message(WARNING "CUDA compiler available but CUDAToolkit not found")
    endif ()
else ()
    message(STATUS "CUDA language not available with current toolchain")
endif ()

check_language(HIP)
if (CMAKE_HIP_COMPILER)
    enable_language(HIP)
    find_package(HIP QUIET)
    if (HIP_FOUND)
        message(STATUS "HIP language enabled and HIP package found")
    else ()
        message(WARNING "HIP compiler available but HIP package not found")
    endif ()
else ()
    message(STATUS "HIP language not available with current toolchain")
endif ()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#######################################################
# Overall Options
#######################################################
include(CMakeDependentOption)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(PPB_LOGGING_LEVEL_LIST "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
set(PPB_LOGGING_LEVEL "INFO" CACHE STRING "Set the Logging level, default (INFO), available options: TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF")
set_property(CACHE PPB_LOGGING_LEVEL PROPERTY STRINGS ${PPB_LOGGING_LEVEL_LIST})
# Convert the logging level string to its corresponding number
list(FIND PPB_LOGGING_LEVEL_LIST ${PPB_LOGGING_LEVEL} PPB_LOGGING_LEVEL_INDEX)
if (${PPB_LOGGING_LEVEL_INDEX} EQUAL -1)
    message(FATAL_ERROR "Invalid logging level: ${PPB_LOGGING_LEVEL}")
endif ()
add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${PPB_LOGGING_LEVEL_INDEX})

option(PPB_ENABLE_OnlyKernelRuntime "Only measures the actual GPU Kernel runtime, but not memory transfers, etc." OFF)

option(PPB_ENABLE_OpenACC "Enable OpenACC Target" OFF)
option(PPB_ENABLE_OpenMP "Enable OpenMP Target" OFF)
option(PPB_ENABLE_AdaptiveCpp "Enable AdaptiveCPP Target" OFF)
option(PPB_ENABLE_Vulkan "Enable Vulkan Kompute Target" OFF)
option(PPB_ENABLE_OpenCL "Enable OpenCL Targets" OFF)
option(PPB_ENABLE_Boost "Enable Boost::Compute Target (requires OpenCL)" OFF)
option(PPB_ENABLE_Kokkos "Enable Kokkos Target" OFF)
option(PPB_ENABLE_Cuda "Enable Cuda and Thrust Targets" ON)
option(PPB_ENABLE_Metal "Enable MetalCpp (Only on macOS!)" OFF)
option(PPB_ENABLE_Alpaka "Enable Alpaka" OFF)

option(PPB_ENABLE_VectorAddition "Enable VectorAddition Benchmarks" OFF)
option(PPB_ENABLE_MatrixMultiplication "Enable MatrixMultiplication Benchmarks" OFF)
option(PPB_ENABLE_NBodySimulation "Enable NBodySimulation Benchmarks" OFF)
option(PPB_ENABLE_Testing "Enable Testing" OFF)

option(PPB_ENABLE_Cuda_DirectSum "Enables the cuda direct sum implementation" ON)
option(PPB_ENABLE_Cuda_LinkedCells "Enables the linked cells implementation of cuda" OFF)
option(PPB_ENABLE_Cuda_VerletLists "Enables the verlet lists implementation of cuda" OFF)

message(STATUS "####################################################")
message(STATUS "Build Type                      ${CMAKE_BUILD_TYPE}")
message(STATUS "Logging Level                   ${PPB_LOGGING_LEVEL}")
message(STATUS "Only Kernel Runtime             ${PPB_ENABLE_OnlyKernelRuntime}")
message(STATUS "####################################################")
message(STATUS "Kokkos                          ${PPB_ENABLE_Kokkos}")
message(STATUS "AdaptiveCpp (needs Boost)       ${PPB_ENABLE_AdaptiveCpp}")
message(STATUS "Vulkan-Kompute                  ${PPB_ENABLE_Vulkan}")
message(STATUS "OpenACC                         ${PPB_ENABLE_OpenACC}")
message(STATUS "OpenMP                          ${PPB_ENABLE_OpenMP}")
message(STATUS "OpenCL                          ${PPB_ENABLE_OpenCL}")
message(STATUS "Boost::Compute (needs OpenCL)   ${PPB_ENABLE_Boost}")
message(STATUS "Cuda                            ${PPB_ENABLE_Cuda}")
message(STATUS "Alpaka                          ${PPB_ENABLE_Alpaka}")
message(STATUS "Metal (Apple)                   ${PPB_ENABLE_Metal}")
message(STATUS "####################################################")
message(STATUS "VectorAddition                  ${PPB_ENABLE_VectorAddition}")
message(STATUS "MatrixMultiplication            ${PPB_ENABLE_MatrixMultiplication}")
message(STATUS "NBodySimulation                 ${PPB_ENABLE_NBodySimulation}")
message(STATUS "Testing                         ${PPB_ENABLE_Testing}")
message(STATUS "####################################################")
message(STATUS "Cuda_DirectSum                  ${PPB_ENABLE_Cuda_DirectSum}")
message(STATUS "Cuda_LinkedCells                ${PPB_ENABLE_Cuda_LinkedCells}")
message(STATUS "Cuda_VerletLists                ${PPB_ENABLE_Cuda_VerletLists}")
message(STATUS "####################################################")

#######################################################
# Including dependencies needed across multiple targets
#######################################################

if (${PPB_ENABLE_OnlyKernelRuntime})
    add_compile_definitions(PPB_MEASURE_ONLY_KERNEL)
endif ()

# Appends the the module path to contain additional CMake modules for this project
# and include everything necessary
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# We always requires Google Benchmark
include(gbench)
include(opencl_compiling)

if (${PPB_ENABLE_Boost} OR ${PPB_ENABLE_AdaptiveCpp})
    # std::random_shuffle was removed in C++17 and deprecated in C++14 - Boost uses it if we don't enable
    # this macro for C++11 usage. Then, Boost uses the modern approach rom the header <random> (compliant with C++17)
    add_compile_definitions(BOOST_COMPUTE_USE_CPP11)
    include(boost)
endif ()

# OpenCL Inclusion
if (${PPB_ENABLE_OpenCL})
    find_package(OpenCL REQUIRED)
endif()

# OpenMP Inclusion
if (${PPB_ENABLE_OpenMP})
    find_package(OpenMP REQUIRED)
endif ()

# OpenACC Inclusion
if (${PPB_ENABLE_OpenACC})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        find_package(OpenACC REQUIRED)
    else ()
        message(FATAL_ERROR "Unsupported Compiler! OpenACC is only supported through the NVIDIA HPC SDK!")
    endif ()
endif ()

# Kokkos Inclusion
if (${PPB_ENABLE_Kokkos})
    include(kokkos)
endif()

# Kokkos Inclusion
if (${PPB_ENABLE_Raja})
    include(raja)
endif()

# AdaptiveCPP Inclusion
if (${PPB_ENABLE_AdaptiveCpp} AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    include(adaptive_cpp)
endif ()

# AdaptiveCPP Inclusion
if (${PPB_ENABLE_Metal})
    include(metal)
endif ()

if (${PPB_ENABLE_Alpaka})
    include(alpaka)
endif ()

if (${PPB_ENABLE_Vulkan})
    include(vulkan)
endif ()

#######################################################
# Including source subdirectories
#######################################################

add_subdirectory(src)

if (${PPB_ENABLE_Testing})
    enable_testing()
    add_subdirectory(test)
endif ()

