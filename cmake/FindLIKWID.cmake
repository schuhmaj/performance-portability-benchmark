# FindLIKWID.cmake - Locate the LIKWID library and headers
#
# This module defines:
#   LIKWID_FOUND          - True if LIKWID is found
#   LIKWID_INCLUDE_DIRS   - The directory containing likwid.h
#   LIKWID_LIBRARIES      - The LIKWID library to link against
#   likwid::likwid        - Imported target
#
# Hints:
#   LIKWID_ROOT           - Root of a LIKWID installation
#   ENV{LIKWID_ROOT}      - Same as above

# Search hints
set(_LIKWID_HINTS
    $ENV{LIKWID_ROOT}
    ${LIKWID_ROOT}
)

find_path(LIKWID_INCLUDE_DIR
    NAMES likwid.h
    HINTS ${_LIKWID_HINTS}
)
find_library(LIKWID_LIBRARY
    NAMES likwid
    HINTS ${_LIKWID_HINTS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIKWID
    REQUIRED_VARS LIKWID_INCLUDE_DIR LIKWID_LIBRARY
    FAIL_MESSAGE "Could not find LIKWID. Set LIKWID_ROOT or ensure likwid.h and liblikwid are available."
)

if(LIKWID_FOUND)
    set(LIKWID_INCLUDE_DIRS "${LIKWID_INCLUDE_DIR}")
    set(LIKWID_LIBRARIES "${LIKWID_LIBRARY}")

    if(NOT TARGET likwid::likwid)
        add_library(likwid::likwid UNKNOWN IMPORTED GLOBAL)
        set_target_properties(likwid::likwid PROPERTIES
            IMPORTED_LOCATION "${LIKWID_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LIKWID_INCLUDE_DIR}"
        )
        target_compile_definitions(likwid::likwid INTERFACE LIKWID_PERFMON)
    endif()
endif()

mark_as_advanced(LIKWID_INCLUDE_DIR LIKWID_LIBRARY)
