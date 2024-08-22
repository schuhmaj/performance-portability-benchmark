cmake_minimum_required(VERSION 3.22)
project(performance_portability_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


#######################################################
# Overall Options
#######################################################
set(LOGGING_LEVEL "2" CACHE STRING "Set the Logging level, default (INFO=2), available options:
TRACE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CRITICAL=5, OFF=6")
set_property(CACHE LOGGING_LEVEL PROPERTY STRINGS 0, 1, 2, 3, 4, 5, 6)
add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${LOGGING_LEVEL})

#######################################################
# Including dependencies needed across multiple targets
#######################################################
# Appends the the module path to contain additional CMake modules for this project
# and include everything necessary
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Include dependecies
include(gbench)
include(spdlog)
include(kokkos)

#######################################################
# Including source subdirectories
#######################################################

add_subdirectory(src)
