cmake_minimum_required(VERSION 3.22)
project(performance_portability_benchmark VERSION 1.0 LANGUAGES CXX)

include(CheckLanguage)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "" FORCE)
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
option(PPB_ENABLE_Vulkan "Enable Vulkan-Kompute Target" OFF)
option(PPB_ENABLE_Slang_Vulkan "Enable Slang-Vulkan Target" OFF)
option(PPB_ENABLE_Slang_Cuda "Enable Slang-Cuda Target" OFF)
option(PPB_ENABLE_OpenCL "Enable OpenCL Targets" OFF)
option(PPB_ENABLE_Boost "Enable Boost::Compute Target (requires OpenCL)" OFF)
option(PPB_ENABLE_Kokkos "Enable Kokkos Target" OFF)
option(PPB_ENABLE_Cuda "Enable Cuda and Thrust Targets" OFF)
option(PPB_ENABLE_Metal "Enable MetalCpp (Only on macOS!)" OFF)
option(PPB_ENABLE_Alpaka "Enable Alpaka" OFF)

option(PPB_ENABLE_VectorAddition "Enable VectorAddition Benchmarks" ON)
option(PPB_ENABLE_MatrixMultiplication "Enable MatrixMultiplication Benchmarks" ON)
option(PPB_ENABLE_NBodySimulation "Enable NBodySimulation Benchmarks" OFF)
option(PPB_ENABLE_PolyhedralGravity "Enable PolyhedralGravity Benchmarks" OFF)
option(PPB_ENABLE_Testing "Enable Testing" ON)

cmake_dependent_option(PPB_ENABLE_NBodySimulation_naive  "Enable naive nBodySimulation"  OFF "PPB_ENABLE_NBodySimulation" OFF)
cmake_dependent_option(PPB_ENABLE_NBodySimulation_cells  "Enable cells nBodySimulation"  OFF "PPB_ENABLE_NBodySimulation" OFF)
cmake_dependent_option(PPB_ENABLE_NBodySimulation_verlet  "Enable verlet nBodySimulation"  OFF "PPB_ENABLE_NBodySimulation" OFF)


set(PPB_Polyhedral_FloatType "32" CACHE STRING "Choose the floating point precision (32 or 64 bits)")
set_property(CACHE PPB_Polyhedral_FloatType PROPERTY STRINGS "32" "64")

message(STATUS "####################################################")
message(STATUS "Build Type                      ${CMAKE_BUILD_TYPE}")
message(STATUS "Logging Level                   ${PPB_LOGGING_LEVEL}")
message(STATUS "Only Kernel Runtime             ${PPB_ENABLE_OnlyKernelRuntime}")
message(STATUS "####################################################")
message(STATUS "Kokkos                          ${PPB_ENABLE_Kokkos}")
message(STATUS "AdaptiveCpp (needs Boost)       ${PPB_ENABLE_AdaptiveCpp}")
message(STATUS "Vulkan-Kompute                  ${PPB_ENABLE_Vulkan}")
message(STATUS "Slang-Vulkan                    ${PPB_ENABLE_Slang_Vulkan}")
message(STATUS "Slang-Cuda                      ${PPB_ENABLE_Slang_Cuda}")
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
message(STATUS "PolyhedralGravity               ${PPB_ENABLE_PolyhedralGravity}")
message(STATUS " ⮑ Polyhedral Float Type       ${PPB_Polyhedral_FloatType}")
message(STATUS "NBodySimulation                 ${PPB_ENABLE_NBodySimulation}")
message(STATUS " ⮑ Naive                       ${PPB_ENABLE_NBodySimulation_naive}")
message(STATUS " ⮑ LinkedCells                 ${PPB_ENABLE_NBodySimulation_cells}")
message(STATUS " ⮑ VerletLists                 ${PPB_ENABLE_NBodySimulation_verlet}")
message(STATUS "####################################################")
message(STATUS "Testing                         ${PPB_ENABLE_Testing}")
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

if (${PPB_ENABLE_Boost} OR ${PPB_ENABLE_AdaptiveCpp})
    # std::random_shuffle was removed in C++17 and deprecated in C++14 - Boost uses it if we don't enable
    # this macro for C++11 usage. Then, Boost uses the modern approach rom the header <random> (compliant with C++17)
    add_compile_definitions(BOOST_COMPUTE_USE_CPP11)
    include(boost)
endif ()

if (${PPB_ENABLE_OpenCL})
    find_package(OpenCL REQUIRED)
    if (APPLE AND TARGET OpenCL::OpenCL)
        # Inject the Homebrew C++ headers into the OpenCL target
        find_path(OpenCL_CXX_INCLUDE_DIR
                NAMES CL/opencl.hpp
                HINTS
                /opt/homebrew/opt/opencl-clhpp-headers/include
                /opt/homebrew/include
                /usr/local/include
                /opt/local/include
        )
        if (OpenCL_CXX_INCLUDE_DIR)
            message(STATUS "Found OpenCL C++ bindings at: ${OpenCL_CXX_INCLUDE_DIR}")
            target_include_directories(OpenCL::OpenCL INTERFACE "${OpenCL_CXX_INCLUDE_DIR}")
        else()
            message(WARNING "Could not find CL/opencl.hpp. Please install opencl-clhpp-headers.")
        endif()
    endif()
    include(compile_opencl)
endif()

if (${PPB_ENABLE_OpenMP})
    find_package(OpenMP REQUIRED)
endif ()

if (${PPB_ENABLE_OpenACC})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        find_package(OpenACC REQUIRED)
    else ()
        message(FATAL_ERROR "Unsupported Compiler! OpenACC is only supported through the NVIDIA HPC SDK!")
    endif ()
endif ()

if (${PPB_ENABLE_Kokkos})
    include(kokkos)
endif()

if (${PPB_ENABLE_Raja})
    include(raja)
endif()

if (${PPB_ENABLE_AdaptiveCpp} AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    include(adaptive_cpp)
endif ()

if (${PPB_ENABLE_Metal})
    include(metal)
endif ()

if (${PPB_ENABLE_Alpaka})
    include(alpaka)
endif ()

# Vulkan Inclusion
if (${PPB_ENABLE_Vulkan})
    include(glm)
    include(compile_vulkan_opengl)
    include(vulkan)
endif ()

if (${PPB_ENABLE_Slang_Cuda} OR ${PPB_ENABLE_Slang_Vulkan})
    include(slang)
endif ()

#######################################################
# Including source subdirectories
#######################################################

add_subdirectory(src)

if (${PPB_ENABLE_Testing})
    enable_testing()
    add_subdirectory(test)
endif ()

