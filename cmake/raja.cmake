include(FetchContent)

message(STATUS "Setting up Raja")

set(RAJA_VERSION 2024.07.0)

include(FetchContent)


set(RAJA_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(RAJA_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAJA_ENABLE_EXERCISES OFF CACHE BOOL "" FORCE)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    FetchContent_Declare(
            Raja
            URL https://github.com/LLNL/RAJA/releases/download/v${RAJA_VERSION}/RAJA-v${RAJA_VERSION}.tar.gz
            FIND_PACKAGE_ARGS
    )
else()
    FetchContent_Declare(
            Raja
            URL https://github.com/LLNL/RAJA/releases/download/v${RAJA_VERSION}/RAJA-v${RAJA_VERSION}.tar.gz
    )
endif()
FetchContent_MakeAvailable(Raja)

if(EXISTS ${CMAKE_BINARY_DIR}/_deps/raja-src)
    message(STATUS "Raja was setup from its GitHub Release ${RAJA_VERSION}")
else()
    message(STATUS "Raja was found on the system!")
endif()