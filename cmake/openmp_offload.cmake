# Provides the INTERFACE target `ppb_openmp_offload` that bundles the compiler
# and linker flags needed for OpenMP target offloading to GPUs. Linking against
# it also pulls in OpenMP::OpenMP_CXX, so it is a drop-in replacement for it.
#
# Supported toolchains:
#   * LLVM/Clang  (CMAKE_CXX_COMPILER_ID == Clang)      -> NVIDIA and AMD GPUs
#   * Intel oneAPI (CMAKE_CXX_COMPILER_ID == IntelLLVM)  -> Intel GPUs
# Any other compiler gets plain host OpenMP (no offload flags are added).
#
# Configure with:
#   -DPPB_OpenMP_Offload_Target=nvidia|amd|intel   (default: nvidia)
#   -DPPB_OpenMP_Offload_Arch=<arch>               (optional, e.g. sm_80, gfx90a)
# When no architecture is given, JIT compilation is used (the GPU arch is
# resolved at run time), which mirrors the previous nvptx64 + -fopenmp-target-jit
# behaviour and keeps a single binary portable across devices of one vendor.

# Pick a sensible default vendor for the configuring machine. This only seeds
# the cache variable's default; an explicit -DPPB_OpenMP_Offload_Target=... wins.
# Priority: Intel compiler -> probed hardware -> detected GPU toolchain -> nvidia.
function(_ppb_detect_omp_offload_target out_var)
    # 1. The Intel oneAPI compiler unambiguously implies an Intel GPU target.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(${out_var} "intel" PARENT_SCOPE)
        return()
    endif ()

    # 2. Probe the hardware actually present on this machine.
    find_program(_ppb_rocminfo NAMES rocminfo)
    mark_as_advanced(_ppb_rocminfo)
    if (_ppb_rocminfo)
        execute_process(COMMAND "${_ppb_rocminfo}"
                RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        if (_rc EQUAL 0)
            set(${out_var} "amd" PARENT_SCOPE)
            return()
        endif ()
    endif ()
    find_program(_ppb_nvidia_smi NAMES nvidia-smi)
    mark_as_advanced(_ppb_nvidia_smi)
    if (_ppb_nvidia_smi)
        execute_process(COMMAND "${_ppb_nvidia_smi}" -L
                RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
        if (_rc EQUAL 0)
            set(${out_var} "nvidia" PARENT_SCOPE)
            return()
        endif ()
    endif ()

    # 3. No GPU visible at configure time (e.g. a login node): fall back to the
    #    toolchain the project already detected, defaulting to nvidia.
    if (CMAKE_HIP_COMPILER AND CMAKE_HIP_PLATFORM STREQUAL "amd")
        set(${out_var} "amd" PARENT_SCOPE)
    else ()
        set(${out_var} "nvidia" PARENT_SCOPE)
    endif ()
endfunction()

_ppb_detect_omp_offload_target(_ppb_omp_target_default)

set(PPB_OpenMP_Offload_Target "${_ppb_omp_target_default}" CACHE STRING
        "GPU vendor for OpenMP target offloading (nvidia, amd, intel)")
set_property(CACHE PPB_OpenMP_Offload_Target PROPERTY STRINGS nvidia amd intel)

set(PPB_OpenMP_Offload_Arch "" CACHE STRING
        "GPU architecture for OpenMP offloading (e.g. sm_80, gfx90a). Empty => JIT")

# Map the chosen vendor to its LLVM OpenMP offload target triple.
if (PPB_OpenMP_Offload_Target STREQUAL "nvidia")
    set(_ppb_omp_triple "nvptx64-nvidia-cuda")
elseif (PPB_OpenMP_Offload_Target STREQUAL "amd")
    set(_ppb_omp_triple "amdgcn-amd-amdhsa")
elseif (PPB_OpenMP_Offload_Target STREQUAL "intel")
    set(_ppb_omp_triple "spir64")
else ()
    message(FATAL_ERROR
            "Unsupported PPB_OpenMP_Offload_Target='${PPB_OpenMP_Offload_Target}' (use nvidia, amd or intel)")
endif ()

# Build the offload flag list shared by compile and link steps. The base
# -fopenmp comes from OpenMP::OpenMP_CXX, so we only add the offload specifics.
set(_ppb_omp_flags -fopenmp-targets=${_ppb_omp_triple})
if (PPB_OpenMP_Offload_Arch)
    list(APPEND _ppb_omp_flags -Xopenmp-target=${_ppb_omp_triple} -march=${PPB_OpenMP_Offload_Arch})
elseif (NOT PPB_OpenMP_Offload_Target STREQUAL "intel")
    # SPIR-V (Intel) is JIT'd by the runtime by default; the flag is Clang-only.
    list(APPEND _ppb_omp_flags -fopenmp-target-jit)
endif ()

# When linking, OpenMP::OpenMP_CXX only pulls in the runtime library, not the
# -fopenmp flag, but -fopenmp-targets requires the base OpenMP flag on the same
# command. Add it explicitly, using the spelling FindOpenMP picked for this
# compiler (-fopenmp for Clang, -fiopenmp for icpx); fall back to -fopenmp.
if (OpenMP_CXX_FLAGS)
    separate_arguments(_ppb_omp_base NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
else ()
    set(_ppb_omp_base -fopenmp)
endif ()
set(_ppb_omp_link_flags ${_ppb_omp_base} ${_ppb_omp_flags})

# rpath to the OpenMP runtime so libomptarget/libomp are found at run time.
# OpenMP_omp_LIBRARY is only set when FindOpenMP links an explicit libomp; with
# AppleClang/libgomp/etc. it is empty, so guard the path query.
if (OpenMP_omp_LIBRARY)
    cmake_path(GET OpenMP_omp_LIBRARY PARENT_PATH _ppb_omp_lib_dir)
    list(APPEND _ppb_omp_link_flags "-Wl,-rpath,${_ppb_omp_lib_dir}")
endif ()

add_library(ppb_openmp_offload INTERFACE)
target_link_libraries(ppb_openmp_offload INTERFACE OpenMP::OpenMP_CXX)

# Only the LLVM-family compilers understand these flags; guard with a generator
# expression so the target stays harmless for GCC/NVHPC (host OpenMP fallback).
set(_ppb_omp_guard "$<CXX_COMPILER_ID:Clang,IntelLLVM>")
target_compile_options(ppb_openmp_offload INTERFACE
        "$<${_ppb_omp_guard}:${_ppb_omp_flags}>")
target_link_options(ppb_openmp_offload INTERFACE
        "$<${_ppb_omp_guard}:${_ppb_omp_link_flags}>")

message(STATUS "OpenMP offload target            ${PPB_OpenMP_Offload_Target} (${_ppb_omp_triple})")
if (PPB_OpenMP_Offload_Arch)
    message(STATUS "OpenMP offload arch              ${PPB_OpenMP_Offload_Arch}")
else ()
    message(STATUS "OpenMP offload arch              JIT (resolved at run time)")
endif ()
