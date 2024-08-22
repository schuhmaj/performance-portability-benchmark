include(FetchContent)

message(STATUS "Setting up Kokkos")

set(KOKKOS_VERSION 4.3.00)

include(FetchContent)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    FetchContent_Declare(
            Kokkos
            URL https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.tar.gz
            FIND_PACKAGE_ARGS
    )
else()
    FetchContent_Declare(
            Kokkos
            URL https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.tar.gz
    )
endif()
FetchContent_MakeAvailable(Kokkos)

if(EXISTS ${CMAKE_BINARY_DIR}/_deps/kokkos-src)
    message(STATUS "Kokkos was setup from its GitHub Release ${KOKKOS_VERSION}")
else()
    message(STATUS "Kokkos was found on the system!")
endif()