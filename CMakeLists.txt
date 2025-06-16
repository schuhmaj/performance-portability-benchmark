cmake_minimum_required(VERSION 3.22)
project(performance_portability_benchmark VERSION 1.0 LANGUAGES CXX)

# Detect platform and conditionally enable CUDA if not Apple
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # Darwin is the system name for macOS
    enable_language(CUDA)
    find_package(CUDAToolkit)
endif ()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#######################################################
# Overall Options
#######################################################
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

option(PPB_ENABLE_OpenACC "Enable OpenACC Target" OFF)
option(PPB_ENABLE_OpenMP "Enable OpenMP Target" OFF)
option(PPB_ENABLE_AdaptiveCpp "Enable AdaptiveCPP Target" OFF)
option(PPB_ENABLE_OpenCL "Enable OpenCL Targets" OFF)
option(PPB_ENABLE_Kokkos "Enable Kokkos Target" OFF)
option(PPB_ENABLE_Cuda "Enable Cuda and Thrust Targets" OFF)
option(PPB_ENABLE_MetalCpp "Enable MetalCpp (Only on macOS!)" OFF)

option(PPB_ENABLE_VectorAddition "Enable VectorAddition Benchmarks" ON)
option(PPB_ENABLE_NBodySimulation "Enable NBodySimulation Benchmarks" OFF)
option(PPB_ENABLE_Testing "Enable Testing" ON)

message(STATUS "####################################################")
message(STATUS "Build Type                      ${CMAKE_BUILD_TYPE}")
message(STATUS "Logging Level                   ${PPB_LOGGING_LEVEL}")
message(STATUS "####################################################")
message(STATUS "Kokkos                          ${PPB_ENABLE_Kokkos}")
message(STATUS "AdaptiveCpp                     ${PPB_ENABLE_AdaptiveCpp}")
message(STATUS "OpenACC                         ${PPB_ENABLE_OpenACC}")
message(STATUS "OpenMP                          ${PPB_ENABLE_OpenMP}")
message(STATUS "OpenCL (incl. Boost::Compute)   ${PPB_ENABLE_OpenCL}")
message(STATUS "Cuda                            ${PPB_ENABLE_Cuda}")
message(STATUS "MetalCpp                        ${PPB_ENABLE_MetalCpp}")
message(STATUS "####################################################")
message(STATUS "VectorAddition                  ${PPB_ENABLE_VectorAddition}")
message(STATUS "NBodySimulation                 ${PPB_ENABLE_NBodySimulation}")
message(STATUS "Testing                         ${PPB_ENABLE_Testing}")
message(STATUS "####################################################")

#######################################################
# Including dependencies needed across multiple targets
#######################################################

# Appends the the module path to contain additional CMake modules for this project
# and include everything necessary
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# We always requires Google Benchmark and Boost Libraries
include(gbench)
include(boost)

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

# Thrust Inclusion (requires Cuda)
if (${PPB_ENABLE_Cuda})
    include(thrust)
    thrust_create_target(Thrust HOST CPP DEVICE CUDA)
endif()

# AdaptiveCPP Inclusion
if (${PPB_ENABLE_AdaptiveCpp})
    include(adaptive_cpp)
endif ()

# AdaptiveCPP Inclusion
if (${PPB_ENABLE_MetalCpp})
    include(metal)
endif ()

if (${PPB_ENABLE_Testing})
    enable_testing()
    add_subdirectory(test)
endif ()

#######################################################
# Including source subdirectories
#######################################################

add_subdirectory(src)
