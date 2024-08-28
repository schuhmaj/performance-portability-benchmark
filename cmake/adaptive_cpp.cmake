# This CMake module file setups AdaptiveCpp
# Ideally, AdaptiveCpp is found on the system
# If this is not the case the module will download AdaptiveCpp from its GitHub Release page
# and will compile and install the project into a subdirectory of the build folder.
# NOTE: This compile/ install steps happens at CMake CONFIGURE Time, since the compiler
# and `acpp` tools need to be present for the configuring the project building on top of it!

set(ADAPTIVE_CPP_VERSION 24.06.0)

# Try to find AdaptiveCpp on the System
find_package(AdaptiveCpp CONFIG)

# This option is exposed. Have a look at AdaptiveCpp for a full overview
option(WITH_SSCP_COMPILER "Enables the `generic` compilation flow using JIT compilation. Requires OpenCL and LLVM library." OFF)

# If not found, then download it and install it during CMake configuration time
if (NOT ${AdaptiveCpp_FOUND})
    message(STATUS "AdaptiveCpp was not found on the system. Trying to install (locally) via CMake...")
    include(FetchContent)

    # Fetch the AdaptiveCpp library during CMake setup
    FetchContent_Declare(
            AdaptiveCpp
            URL https://github.com/AdaptiveCpp/AdaptiveCpp/archive/refs/tags/v${ADAPTIVE_CPP_VERSION}.tar.gz
    )
    FetchContent_GetProperties(AdaptiveCpp)
    if(NOT AdaptiveCpp_POPULATED)
        # Fetch the content using previously declared details
        FetchContent_Populate(AdaptiveCpp)
    endif ()
    # We do not call add_subdirectory here, as we don't want acpp in our CMake setup

    # Configure AdaptiveCpp. These are respectively the "build" directory inside the calling CMake build folder
    # and the "install" directory which is located in the calling CMake build folder as well
    set(ADAPTIVE_CPP_BINARY_DIR ${CMAKE_BINARY_DIR}/AdaptiveCpp-build)
    set(ADAPTIVE_CPP_INSTALL_DIR ${CMAKE_BINARY_DIR}/AdaptiveCpp-install)

    # Ensure the binary and install directories exist
    file(MAKE_DIRECTORY ${ADAPTIVE_CPP_BINARY_DIR})
    file(MAKE_DIRECTORY ${ADAPTIVE_CPP_INSTALL_DIR})

    # Set-Up the CMake project in the above specified "build" directory
    execute_process(
            COMMAND ${CMAKE_COMMAND} 
                -G "${CMAKE_GENERATOR}" -S ${adaptivecpp_SOURCE_DIR} -B ${ADAPTIVE_CPP_BINARY_DIR}
                -DCMAKE_INSTALL_PREFIX=${ADAPTIVE_CPP_INSTALL_DIR}
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} 
                -DWITH_SSCP_COMPILER=${WITH_SSCP_COMPILER}
            RESULT_VARIABLE result
    )
    if (result)
        message(FATAL_ERROR "CMake step for AdaptiveCpp failed: ${result}")
    endif ()

    # Build and install AdaptiveCpp into the above specified "local" directory
    execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${ADAPTIVE_CPP_BINARY_DIR} --config Release --target install
            RESULT_VARIABLE result
    )
    if (result)
        message(FATAL_ERROR "Build step for AdaptiveCpp failed: ${result}")
    endif ()

    message(STATUS "AdaptiveCpp was successfully installed at ${ADAPTIVE_CPP_INSTALL_DIR}!")

    # Add the installation path to CMake's search paths and retry to find_package
    list(APPEND CMAKE_PREFIX_PATH ${ADAPTIVE_CPP_INSTALL_DIR})
    find_package(AdaptiveCpp CONFIG REQUIRED)
else ()
    message(STATUS "AdaptiveCPP was found at ${AdaptiveCpp_DIR}!")
endif ()

# Set debug level of ACPP based on build type and
# other add some compile definitions given by the AdaptiveCpp example
# see, https://github.com/AdaptiveCpp/AdaptiveCpp/blob/fc51dae9006d6858fc9c33148cc5f935bb56b075/examples/CMakeLists.txt#L17
if (NOT ACPP_DEBUG_LEVEL)
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        set(ACPP_DEBUG_LEVEL 3 CACHE STRING
                "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                FORCE)
    else ()
        set(ACPP_DEBUG_LEVEL 2 CACHE STRING
                "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                FORCE)
    endif ()
endif ()

add_compile_definitions(HIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})

if (WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
endif ()
