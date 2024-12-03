include(FetchContent)

message(STATUS "Setting up Boost")

set(BOOST_VERSION 1.85.0)

cmake_policy(SET CMP0167 NEW)
find_package(Boost QUIET)

if (${Boost_FOUND})
    message(STATUS "Found existing boost libraries: ${Boost_DIR}")
    set(BOOST_DOWNLOADED False)
else ()
    message(STATUS "Using Boost from GitHub Release ${BOOST_VERSION}")
    include(FetchContent)
    FetchContent_Declare(
            Boost
            URL https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}-cmake.tar.gz
    )
    FetchContent_MakeAvailable(Boost)
    set(Boost_ROOT ${Boost_SOURCE_DIR})
endif ()