# Toolchain file for IntelLLVM (icx/icpx): pin the C++ standard library to the
# GCC currently in PATH (e.g. a loaded `gcc/14` module) and bake that GCC's lib
# dir into the executable RPATH. This removes the runtime error
#     libstdc++.so.6: version `GLIBCXX_3.4.xx' not found
# with NO manual environment setup.
#
# Unlike CMakePresets (which can expand $env{...} but cannot run commands), a
# toolchain file may invoke gcc/g++ via execute_process, so the paths are
# derived here automatically instead of being exported or hardcoded.

find_program(_PPB_GCC gcc)
find_program(_PPB_GXX g++)

if (_PPB_GCC AND _PPB_GXX)
    # --gcc-install-dir wants .../lib/gcc/<triple>/<version> (dir of crtbegin.o)
    execute_process(
        COMMAND "${_PPB_GCC}" -print-file-name=crtbegin.o
        OUTPUT_VARIABLE _ppb_crtbegin
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(_ppb_gcc_install_dir "${_ppb_crtbegin}" DIRECTORY)

    # RPATH wants the lib64 holding libstdc++.so.6 (real path, no '..' segments)
    execute_process(
        COMMAND "${_PPB_GXX}" -print-file-name=libstdc++.so
        OUTPUT_VARIABLE _ppb_libstdcxx
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(_ppb_lib_dir "${_ppb_libstdcxx}" DIRECTORY)
    file(REAL_PATH "${_ppb_lib_dir}" _ppb_gcc_lib_dir)

    # gcc echoes the bare filename back when it can't locate the file, so only
    # proceed when both lookups returned absolute paths.
    if (IS_ABSOLUTE "${_ppb_crtbegin}" AND IS_ABSOLUTE "${_ppb_libstdcxx}")
        set(_ppb_flags "--gcc-install-dir=${_ppb_gcc_install_dir} -Wl,-rpath=${_ppb_gcc_lib_dir}")
        string(APPEND CMAKE_C_FLAGS_INIT   " ${_ppb_flags}")
        string(APPEND CMAKE_CXX_FLAGS_INIT " ${_ppb_flags}")
        message(STATUS "intel-gcc-toolchain: pinned libstdc++ to ${_ppb_gcc_install_dir}")
    else ()
        message(WARNING "intel-gcc-toolchain: could not locate a GCC libstdc++ "
                        "(is a modern gcc module loaded?); flags not applied.")
    endif ()
else ()
    message(WARNING "intel-gcc-toolchain: gcc/g++ not found in PATH; flags not applied.")
endif ()
