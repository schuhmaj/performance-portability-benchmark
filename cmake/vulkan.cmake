message(STATUS "Setting up Vulkan Kompute")
set(Kompute_VERSION 0.9.0)

find_package(Vulkan REQUIRED)
find_package(kompute ${Kompute_VERSION} QUIET)

if (${kompute_FOUND})
    message(STATUS "Found existing Vulkan Kompute libraries: ${kompute_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${kompute_DIR}")
else ()
    message(STATUS "Using Vulkan Kompute from GitHub Release ${Kompute_VERSION}")
    include(FetchContent)

    FetchContent_Declare(
            Kompute
            URL https://github.com/KomputeProject/kompute/archive/refs/tags/v${Kompute_VERSION}.tar.gz
    )

    set(KOMPUTE_OPT_LOG_LEVEL "Off" CACHE STRING "Kompute log level" FORCE)

    FetchContent_MakeAvailable(Kompute)
    target_compile_options(kompute PRIVATE
            -Wno-error
            -Wno-deprecated-literal-operator
            -Wno-deprecated
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-sign-compare
    )
endif ()

list(APPEND CMAKE_PREFIX_PATH "${kompute_SOURCE_DIR}/cmake")

# function for compiling vulkan shaders in a directory using kompute
# note
#  - all files in the specified directory are expected to be .comp files
function(vulkan_compile_shaders_path DIR_PATH OUT_HEADERS NAMESPACE)
    file(GLOB files CONFIGURE_DEPENDS "${DIR_PATH}/*")
    set(HEADERS "")

    foreach(file IN LISTS files)
        get_filename_component(kernel "${file}" NAME_WE)
        set(COMP_FILE ${kernel}.comp)
        set(OUT_HEADER ${kernel}.h)

        vulkan_compile_shader(
            INFILE shaders/${COMP_FILE}
            OUTFILE ${OUT_HEADER}
            NAMESPACE ${NAMESPACE}
            RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake"
        )
        list(APPEND HEADERS "${CMAKE_CURRENT_BINARY_DIR}/${OUT_HEADER}")
    endforeach()

    set(${OUT_HEADERS} "${HEADERS}" PARENT_SCOPE)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${OUT_HEADER}
        PROPERTIES GENERATED TRUE
    )

endfunction()