# Provides the INTERFACE target `ppb_stdpar_offload` that bundles the compiler
# and linker flags needed to offload ISO C++ standard parallel algorithms
# (stdpar, i.e. <execution> + std::execution::par_unseq) to GPUs.
#
# Supported vendor toolchains (selected automatically via CMAKE_CXX_COMPILER_ID):
#   * NVIDIA HPC SDK (CMAKE_CXX_COMPILER_ID == NVHPC)      -> -stdpar=gpu
#   * ROCm amdclang++ >= 6.1 (CMAKE_CXX_COMPILER_ID == Clang) -> --hipstdpar
#   * Intel oneAPI icpx >= 2024.0 (CMAKE_CXX_COMPILER_ID == IntelLLVM) -> -fsycl-pstl-offload=gpu
# Any other compiler gets a host fallback: no offload flags, but TBB is linked
# so that libstdc++'s parallel STL backend still parallelizes on the CPU.
#
# Alternatively, AdaptiveCpp's stdpar offloading can be used INSTEAD of the
# vendor toolchain by configuring with
#   -DPPB_Stdpar_UseAcpp=ON
# This requires that acpp is the C++ compiler (CMAKE_CXX_COMPILER=acpp); the
# device targets are then controlled through acpp itself (--acpp-targets,
# default: the portable `generic` SSCP flow).
#
# Configure with:
#   -DPPB_Stdpar_Offload_Arch=<arch>   (optional; cc80/native for NVHPC,
#                                       gfx90a/native for amdclang++,
#                                       ignored for Intel and AdaptiveCpp)
#
# NOTE: All memory that is accessed inside offloaded algorithms must come from
# heap allocations of translation units compiled with these flags (unified /
# managed memory interposition). Linking this target PUBLIC into the
# implementation libraries propagates the flags to the final executables.

option(PPB_Stdpar_UseAcpp
        "Use AdaptiveCpp (acpp) stdpar offloading instead of the vendor toolchain (requires CMAKE_CXX_COMPILER=acpp)" OFF)

set(PPB_Stdpar_Offload_Arch "" CACHE STRING
        "GPU arch for stdpar offload (NVHPC: cc80/native, AMD: gfx90a/native). Empty => toolchain default (AMD: native)")

option(PPB_Stdpar_Hip_InterposeAlloc
        "Add --hipstdpar-interpose-alloc for systems without HMM/XNACK support (AMD only)" OFF)

option(PPB_Stdpar_Acpp_UnconditionalOffload
        "Always offload stdpar algorithms with AdaptiveCpp instead of relying on its heuristics" ON)

set(PPB_Stdpar_Intel_OffloadTarget "gpu" CACHE STRING
        "Device -fsycl-pstl-offload targets (Intel only). Set to 'cpu' to check whether a wrong result \
comes from the kernels or from the GPU driver stack.")
set_property(CACHE PPB_Stdpar_Intel_OffloadTarget PROPERTY STRINGS "gpu" "cpu")

add_library(ppb_stdpar_offload INTERFACE)

if (PPB_Stdpar_UseAcpp)
    # AdaptiveCpp: the acpp driver (identifying itself as Clang) rewrites the
    # standard algorithms onto its SYCL runtime. The device targets follow the
    # acpp configuration (--acpp-targets), so no vendor handling is needed here.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        message(FATAL_ERROR
                "PPB_Stdpar_UseAcpp=ON requires acpp as the C++ compiler "
                "(CMAKE_CXX_COMPILER=acpp), but got '${CMAKE_CXX_COMPILER_ID}'")
    endif ()
    set(_ppb_stdpar_flags --acpp-stdpar)
    if (PPB_Stdpar_Acpp_UnconditionalOffload)
        list(APPEND _ppb_stdpar_flags --acpp-stdpar-unconditional-offload)
    endif ()
    set(_ppb_stdpar_link_flags ${_ppb_stdpar_flags})
    set(_ppb_stdpar_backend "AdaptiveCpp (--acpp-stdpar)")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    # NVHPC: -stdpar=gpu offloads the parallel algorithms and turns all heap
    # allocations into CUDA managed memory. Without an explicit arch, nvc++
    # targets the GPU visible on the build machine.
    set(_ppb_stdpar_flags -stdpar=gpu)
    if (PPB_Stdpar_Offload_Arch)
        list(APPEND _ppb_stdpar_flags -gpu=${PPB_Stdpar_Offload_Arch})
    endif ()
    set(_ppb_stdpar_link_flags ${_ppb_stdpar_flags})
    set(_ppb_stdpar_backend "NVHPC (-stdpar=gpu)")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Intel oneAPI: PSTL offload maps the standard algorithms onto oneDPL/SYCL.
    # SPIR-V is JIT'd by the runtime, so no architecture flag applies.
    #
    # -fsycl-pstl-offload only exists from oneAPI 2024.0 on; older icpx (e.g. the 2023.x
    # release that matches a 2023 Level Zero/OpenCL driver stack) rejects it outright with
    # "unknown argument". There is no stdpar offload path for those compilers, so fail early
    # with an actionable message instead of letting the build die in the first source file.
    #
    # NOTE: this is deliberately a version comparison and not a check_cxx_compiler_flag() probe.
    # The flag makes the driver force-include pstl_offload_impl.h, which drags in <sycl/sycl.hpp>
    # and the oneDPL headers; the bare try_compile of the probe has neither the project's include
    # paths nor its C++ standard, so it reports a false negative on compilers that do support the
    # flag. IntelLLVM reports the oneAPI release as its version, so the comparison is exact.
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2024.0)
        message(FATAL_ERROR
                "icpx ${CMAKE_CXX_COMPILER_VERSION} does not support "
                "-fsycl-pstl-offload=${PPB_Stdpar_Intel_OffloadTarget} "
                "(added in oneAPI 2024.0), so stdpar cannot be offloaded with this toolchain. Either\n"
                "  * use a oneAPI >= 2024.0 compiler, or\n"
                "  * build the stdpar targets through AdaptiveCpp instead "
                "(-DCMAKE_CXX_COMPILER=acpp -DPPB_Stdpar_UseAcpp=ON), which reaches Intel GPUs via "
                "its own SPIR-V/Level Zero path and does not depend on the oneAPI PSTL offload flag, or\n"
                "  * configure with -DPPB_ENABLE_Stdpar=OFF to skip the stdpar targets.")
    endif ()
    set(_ppb_stdpar_flags -fsycl -fsycl-pstl-offload=${PPB_Stdpar_Intel_OffloadTarget})
    set(_ppb_stdpar_link_flags ${_ppb_stdpar_flags})
    set(_ppb_stdpar_backend "Intel oneAPI (-fsycl-pstl-offload=${PPB_Stdpar_Intel_OffloadTarget})")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # ROCm amdclang++ (>= ROCm 6.1) or a matching upstream Clang: --hipstdpar
    # offloads par_unseq algorithms via rocThrust. An offload arch is mandatory,
    # so default to `native` (the GPU visible on the build machine).
    # NOTE: Upstream Clang must be new enough for the ROCm headers on the system
    # (ROCm 7.x hipstdpar headers need LLVM >= 21; older upstream Clang fails
    # with opencl_private/address_space(5) mismatches - use amdclang++ instead).
    if (PPB_Stdpar_Offload_Arch)
        set(_ppb_stdpar_arch ${PPB_Stdpar_Offload_Arch})
    else ()
        set(_ppb_stdpar_arch native)
    endif ()
    set(_ppb_stdpar_flags --hipstdpar --offload-arch=${_ppb_stdpar_arch})
    if (PPB_Stdpar_Hip_InterposeAlloc)
        # Replaces every heap allocation with hipMallocManaged; required on
        # systems whose GPUs lack HMM/XNACK-enabled page migration.
        list(APPEND _ppb_stdpar_flags --hipstdpar-interpose-alloc)
    endif ()
    set(_ppb_stdpar_link_flags --hipstdpar)
    set(_ppb_stdpar_backend "ROCm hipstdpar (--hipstdpar, arch: ${_ppb_stdpar_arch})")
    if (NOT PPB_Stdpar_Hip_InterposeAlloc)
        # Without allocation interposition the GPU must page in ordinary host
        # heap memory on demand, which requires XNACK (disabled by default on
        # most systems, incl. MI200/MI300). Otherwise the binaries die with
        # 'Memory access fault by GPU node-N'.
        message(STATUS "Stdpar hipstdpar note            run binaries with HSA_XNACK=1 "
                "(or configure -DPPB_Stdpar_Hip_InterposeAlloc=ON if XNACK is unavailable)")
    endif ()
else ()
    # Host fallback: keep the targets buildable, executing on the CPU. libstdc++
    # dispatches the parallel algorithms to TBB, so link it explicitly.
    message(WARNING
            "No stdpar GPU offload support for compiler '${CMAKE_CXX_COMPILER_ID}' "
            "(use NVHPC, ROCm amdclang++, Intel icpx, or acpp with PPB_Stdpar_UseAcpp=ON). "
            "The stdpar targets fall back to host execution.")
    set(_ppb_stdpar_flags "")
    set(_ppb_stdpar_link_flags "")
    set(_ppb_stdpar_backend "host fallback (TBB)")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # libc++ ships its parallel STL (std::execution policies) only behind
        # the experimental-library flag.
        set(_ppb_stdpar_flags -fexperimental-library)
        set(_ppb_stdpar_link_flags -fexperimental-library)
    endif ()
    include(tbb)
    target_link_libraries(ppb_stdpar_offload INTERFACE TBB::tbb)
endif ()

if (_ppb_stdpar_flags)
    target_compile_options(ppb_stdpar_offload INTERFACE ${_ppb_stdpar_flags})
endif ()
if (_ppb_stdpar_link_flags)
    target_link_options(ppb_stdpar_offload INTERFACE ${_ppb_stdpar_link_flags})
endif ()

message(STATUS "Stdpar offload backend           ${_ppb_stdpar_backend}")
