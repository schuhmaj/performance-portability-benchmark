# Provides the INTERFACE target `ppb_sycl_offload` that bundles the `-fsycl`
# compiler and linker flags needed to offload SYCL kernels to a GPU with the
# Intel oneAPI (DPC++ / IntelLLVM) compiler. Linking against it adds both the
# compile and the link flags, so it is a drop-in for the SYCL device target
# selection.
#
# This mirrors `ppb_openmp_offload` (cmake/openmp_offload.cmake): the GPU vendor
# and (optional) architecture are configurable instead of being hard-wired to
# nvptx64. With no architecture given, JIT compilation is used so a single
# binary stays portable across the devices of one vendor.
#
# Only meaningful for the IntelLLVM (icpx) compiler; this module is included
# only in that case. AdaptiveCpp / other SYCL implementations use
# add_sycl_to_target instead and never see this target.
#
# Configure with:
#   -DPPB_SYCL_Offload_Target=nvidia|amd|intel   (default: intel)
#   -DPPB_SYCL_Offload_Arch=<arch>               (optional, e.g. sm_80, gfx90a)
# When no architecture is given, device code is JIT-compiled at run time. For
# Intel GPUs JIT (spir64) is always used and the arch is ignored, because
# ahead-of-time compilation would require a matching `ocloc` at build time.

set(PPB_SYCL_Offload_Target "intel" CACHE STRING
        "GPU vendor for SYCL offload with IntelLLVM (nvidia, amd, intel)")
set_property(CACHE PPB_SYCL_Offload_Target PROPERTY STRINGS nvidia amd intel)

set(PPB_SYCL_Offload_Arch "" CACHE STRING
        "GPU arch for AOT SYCL offload (e.g. sm_80, gfx90a). Empty => JIT. Ignored for intel (always spir64 JIT)")

# Map the chosen vendor to its JIT target triple and its AOT device-alias prefix.
if (PPB_SYCL_Offload_Target STREQUAL "nvidia")
    set(_ppb_sycl_triple "nvptx64-nvidia-cuda")
    set(_ppb_sycl_alias_prefix "nvidia_gpu_")
elseif (PPB_SYCL_Offload_Target STREQUAL "amd")
    set(_ppb_sycl_triple "amdgcn-amd-amdhsa")
    set(_ppb_sycl_alias_prefix "amd_gpu_")
elseif (PPB_SYCL_Offload_Target STREQUAL "intel")
    set(_ppb_sycl_triple "spir64")
    set(_ppb_sycl_alias_prefix "intel_gpu_")
else ()
    message(FATAL_ERROR
            "Unsupported PPB_SYCL_Offload_Target='${PPB_SYCL_Offload_Target}' (use nvidia, amd or intel)")
endif ()

# Decide the -fsycl-targets value.
if (PPB_SYCL_Offload_Target STREQUAL "intel")
    # SPIR-V is JIT'd by the Level Zero runtime itself; AOT would need a matching
    # `ocloc` at build time, so we always stay on the generic spir64 target.
    set(_ppb_sycl_targets "spir64")
elseif (PPB_SYCL_Offload_Arch)
    # AOT: the device alias (e.g. nvidia_gpu_sm_80, amd_gpu_gfx90a) fully compiles
    # the device code at build time and selects the matching backend implicitly.
    set(_ppb_sycl_targets "${_ppb_sycl_alias_prefix}${PPB_SYCL_Offload_Arch}")
else ()
    # JIT: emit the generic triple and lower it at run time. Portable across one
    # vendor's devices (NVIDIA needs ptxas on PATH; AMD usually requires an
    # explicit arch, i.e. set PPB_SYCL_Offload_Arch).
    set(_ppb_sycl_targets "${_ppb_sycl_triple}")
endif ()

set(_ppb_sycl_flags -fsycl -fsycl-targets=${_ppb_sycl_targets})

add_library(ppb_sycl_offload INTERFACE)

# Only IntelLLVM understands -fsycl; guard with a generator expression so the
# target stays harmless if it ever ends up on a non-IntelLLVM target.
set(_ppb_sycl_guard "$<CXX_COMPILER_ID:IntelLLVM>")
target_compile_options(ppb_sycl_offload INTERFACE
        "$<${_ppb_sycl_guard}:${_ppb_sycl_flags}>")
target_link_options(ppb_sycl_offload INTERFACE
        "$<${_ppb_sycl_guard}:${_ppb_sycl_flags}>")

message(STATUS "SYCL offload target              ${PPB_SYCL_Offload_Target} (${_ppb_sycl_targets})")
if (PPB_SYCL_Offload_Arch AND NOT PPB_SYCL_Offload_Target STREQUAL "intel")
    message(STATUS "SYCL offload arch                ${PPB_SYCL_Offload_Arch}")
else ()
    message(STATUS "SYCL offload arch                JIT (resolved at run time)")
endif ()
