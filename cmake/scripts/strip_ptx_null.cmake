# strip_ptx_null.cmake - Build-time post-process for Slang-generated PTX.
#
# Fixes two issues with slangc's PTX output that break modern CUDA drivers:
#
# 1. Trailing NUL (0x00) byte at end of file.
#    Newer CUDA driver JITs (CUDA 13.x) reject PTX containing NULs with
#    CUDA_ERROR_INVALID_PTX (often surfaces later at cuLaunchKernel).
#    ptxas reports "Unexpected EOF".
#
# 2. `.version` directive set to the slangc toolchain's PTX ISA (e.g. 9.3 for
#    a CUDA 13.3-based slangc), which is rejected by drivers that only
#    support older PTX ISAs with:
#      "the provided PTX was compiled with an unsupported toolchain"
#    (i.e. CUDA_ERROR_UNSUPPORTED_PTX_VERSION). The PTX emitted by Slang for
#    these compute kernels uses only long-stable instructions, so we lower
#    .version to a value that any reasonably recent driver supports while
#    still covering current GPU architectures via JIT forward-compilation.
#
# CMake's string operations cannot match against NUL (C strings terminate at
# the first NUL), so we use `tr` for the NUL strip and `sed` for the version
# rewrite — the build targets Linux/macOS.

if (NOT DEFINED INPUT_FILE)
    message(FATAL_ERROR "INPUT_FILE is required")
endif ()

# PTX 8.7 (CUDA 12.7) is broadly supported by current drivers and is
# sufficient to be JIT'd up to Blackwell (sm_120) and beyond.
if (NOT DEFINED PTX_VERSION)
    set(PTX_VERSION "8.7")
endif ()

set(_tmp "${INPUT_FILE}.fix")

# Step 1: strip NUL bytes.
execute_process(
        COMMAND tr -d "\\000"
        INPUT_FILE  "${INPUT_FILE}"
        OUTPUT_FILE "${_tmp}"
        RESULT_VARIABLE _res
)
if (NOT _res EQUAL 0)
    message(FATAL_ERROR "strip_ptx_null: tr failed (code ${_res}) on ${INPUT_FILE}")
endif ()

# Step 2: lower the .version directive in-place.
execute_process(
        COMMAND sed -i "s/^\\.version[ \\t]\\+[0-9]\\+\\.[0-9]\\+/.version ${PTX_VERSION}/" "${_tmp}"
        RESULT_VARIABLE _res
)
if (NOT _res EQUAL 0)
    message(FATAL_ERROR "strip_ptx_null: sed failed (code ${_res}) on ${_tmp}")
endif ()

file(RENAME "${_tmp}" "${INPUT_FILE}")
