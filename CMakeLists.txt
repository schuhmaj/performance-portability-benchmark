cmake_minimum_required(VERSION 3.22)
project(performance_portability_benchmark VERSION 1.0 LANGUAGES CXX)

# Detect platform and conditionally enable CUDA if not Apple
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # Darwin is the system name for macOS
    enable_language(CUDA)
    find_package(CUDAToolkit)
    message(STATUS "CUDA support enabled")
else()
    message(STATUS "CUDA support not enabled on Apple platforms")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#######################################################
# Overall Options
#######################################################
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Building in CMAKE_BUILD_TYPE = '${CMAKE_BUILD_TYPE}' mode")

set(LOGGING_LEVEL "2" CACHE STRING "Set the Logging level, default (INFO=2), available options:
TRACE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CRITICAL=5, OFF=6")
set_property(CACHE LOGGING_LEVEL PROPERTY STRINGS 0, 1, 2, 3, 4, 5, 6)
add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${LOGGING_LEVEL})

option(PPB_ENABLE_OpenACC "Enable OpenACC Target" OFF)
option(PPB_ENABLE_OpenMP "Enable OpenMP Target" OFF)
option(PPB_ENABLE_AdaptiveCpp "Enable AdaptiveCPP Target" OFF)
option(PPB_ENABLE_OpenCL "Enable AdaptiveCPP Target" OFF)

#######################################################
# Including dependencies needed across multiple targets
#######################################################

if (${PPB_ENABLE_OpenCL})
    find_package(OpenCL REQUIRED)
endif()

if (${PPB_ENABLE_OpenMP})
    find_package(OpenMP REQUIRED)
endif ()

if (${PPB_ENABLE_OpenACC})
    find_package(OpenACC REQUIRED)
endif ()

# Appends the the module path to contain additional CMake modules for this project
# and include everything necessary
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Include dependecies
include(gbench)
include(kokkos)
include(boost)
include(thrust)

if (${PPB_ENABLE_AdaptiveCpp})
    include(adaptive_cpp)
endif ()

thrust_create_target(Thrust HOST CPP DEVICE CUDA)


#######################################################
# Including source subdirectories
#######################################################

add_subdirectory(src)
