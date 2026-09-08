#pragma once

/**
 * @file Marker.h
 *
 * A paradigm-agnostic way to name the regions a profiler should report on.
 *
 * A region encloses exactly the work the benchmark already times, and carries a
 * tag naming what that work is ("matmul", "init", "evaluate"). Two independent
 * back-ends consume those tags, and each is enabled by its own CMake option:
 *
 *   PPB_ENABLE_LIKWID   reads hardware counters inside the region
 *   PPB_ENABLE_NVTX     names the region for Nsight Compute / Nsight Systems
 *
 * Both may be on at once, and with neither one every macro expands to nothing,
 * so the normal benchmark build is untouched.
 *
 * An implementation states where the enclosed work runs, because LIKWID has two
 * separate marker APIs -- LIKWID_MARKER_* reads CPU counters, NVMON_MARKER_*
 * reads Nvidia GPU counters through CUPTI -- and the choice is not a property of
 * the compiler driving a translation unit but of where the kernel executes:
 *
 *   PPB_MARKER_GPU_START / PPB_MARKER_GPU_STOP   device kernels
 *   PPB_MARKER_CPU_START / PPB_MARKER_CPU_STOP   host kernels
 *
 * NVTX does not make that distinction: a range is a host-side annotation either
 * way, and a profiler attributes to it whatever the thread launched while the
 * range was open.
 *
 * @note LIKWID tags have to be unique within an executable, because LIKWID
 * accumulates every entry of a region into one record; two implementations
 * sharing a tag -- as the CUDA matrix multiplications do, being linked into one
 * binary -- would report their counters merged. NVTX has no such constraint,
 * but the same tags serve both, so they stay unique.
 *
 * @note NVMON reads the GPU counters through CUPTI when the region ends, which
 * requires two things. The enclosed work must have completed, so a region
 * around an asynchronous launch needs a synchronization before it closes;
 * PPB_MARKER_SYNC_REGION guards the extra call. And CUPTI needs a current CUDA
 * context, so
 * only paradigms that reach the GPU through CUDA (CUDA, HIP-on-NVIDIA, Kokkos,
 * RAJA, Alpaka, AdaptiveCpp, stdpar, OpenACC, OpenMP target, Slang-CUDA) can be
 * measured that way. OpenCL and Vulkan build their own contexts and stay
 * invisible. NVTX has no such limitation -- the ranges are pure host-side
 * annotations -- which is why Nsight Systems can name an OpenCL or Vulkan
 * region that LIKWID and Nsight Compute cannot even see.
 */

#ifdef PPB_ENABLE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef PPB_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#if defined(PPB_ENABLE_LIKWID) || defined(PPB_ENABLE_NVTX)
/** Whether any region back-end is compiled in. */
#define PPB_MARKER_ENABLED 1
#endif

/* --------------------------------------------------------------------------
 * Per-back-end primitives. Each pair expands to nothing when its back-end is
 * off, so the composed macros below need no further conditionals.
 * ------------------------------------------------------------------------ */

#ifdef PPB_ENABLE_LIKWID
#define PPB_LIKWID_INIT           \
    do {                          \
        LIKWID_MARKER_INIT;       \
        LIKWID_MARKER_THREADINIT; \
        NVMON_MARKER_INIT;        \
    } while (0)
#define PPB_LIKWID_CLOSE     \
    do {                     \
        NVMON_MARKER_CLOSE;  \
        LIKWID_MARKER_CLOSE; \
    } while (0)
#define PPB_LIKWID_GPU_START(tag) NVMON_MARKER_START(tag)
#define PPB_LIKWID_GPU_STOP(tag) NVMON_MARKER_STOP(tag)
#define PPB_LIKWID_CPU_START(tag) LIKWID_MARKER_START(tag)
#define PPB_LIKWID_CPU_STOP(tag) LIKWID_MARKER_STOP(tag)
#else
#define PPB_LIKWID_INIT
#define PPB_LIKWID_CLOSE
#define PPB_LIKWID_GPU_START(tag)
#define PPB_LIKWID_GPU_STOP(tag)
#define PPB_LIKWID_CPU_START(tag)
#define PPB_LIKWID_CPU_STOP(tag)
#endif

#ifdef PPB_ENABLE_NVTX
#define PPB_NVTX_PUSH(tag) nvtxRangePushA(tag)
#define PPB_NVTX_POP() nvtxRangePop()
#else
#define PPB_NVTX_PUSH(tag)
#define PPB_NVTX_POP()
#endif

/* --------------------------------------------------------------------------
 * The macros the implementations use.
 * ------------------------------------------------------------------------ */

#define PPB_MARKER_INIT PPB_LIKWID_INIT
#define PPB_MARKER_CLOSE PPB_LIKWID_CLOSE

#define PPB_MARKER_GPU_START(tag)   \
    do {                            \
        PPB_LIKWID_GPU_START(tag);  \
        PPB_NVTX_PUSH(tag);         \
    } while (0)
#define PPB_MARKER_GPU_STOP(tag)   \
    do {                           \
        PPB_NVTX_POP();            \
        PPB_LIKWID_GPU_STOP(tag);  \
    } while (0)
#define PPB_MARKER_CPU_START(tag)   \
    do {                            \
        PPB_LIKWID_CPU_START(tag);  \
        PPB_NVTX_PUSH(tag);         \
    } while (0)
#define PPB_MARKER_CPU_STOP(tag)   \
    do {                           \
        PPB_NVTX_POP();            \
        PPB_LIKWID_CPU_STOP(tag);  \
    } while (0)

#ifdef PPB_MARKER_ENABLED

/**
 * Guards the synchronization a marked asynchronous region needs.
 *
 * Both back-ends need the enclosed work to have *happened* inside the region,
 * not merely to have been submitted from it: LIKWID reads the GPU counters when
 * the region closes, and Nsight Systems attributes a sample to a range by
 * timestamp. A region around an asynchronous launch therefore waits for it
 * before closing. The wait only exists in a marker build, so the measuring
 * build keeps its overlapping submissions.
 */
#define PPB_MARKER_SYNC_REGION 1

/**
 * Scope guards for regions that span a whole function. The enclosed work is
 * complete once the function is ready to return -- every implementation reads
 * its result back to the host -- so no extra synchronization is needed here.
 */
namespace ppb::marker {
    template<bool OnDevice>
    class Region {
    public:
        explicit Region(const char *tag) : _tag(tag) {
            if constexpr (OnDevice) {
                PPB_MARKER_GPU_START(_tag);
            } else {
                PPB_MARKER_CPU_START(_tag);
            }
        }
        ~Region() {
            if constexpr (OnDevice) {
                PPB_MARKER_GPU_STOP(_tag);
            } else {
                PPB_MARKER_CPU_STOP(_tag);
            }
        }
        Region(const Region &) = delete;
        Region &operator=(const Region &) = delete;

    private:
        const char *_tag;
    };
}// namespace ppb::marker

#define PPB_MARKER_GPU_SCOPE(tag) const ::ppb::marker::Region<true> ppbMarkerRegion_(tag)
#define PPB_MARKER_CPU_SCOPE(tag) const ::ppb::marker::Region<false> ppbMarkerRegion_(tag)

#else /* PPB_MARKER_ENABLED */

#define PPB_MARKER_GPU_SCOPE(tag)
#define PPB_MARKER_CPU_SCOPE(tag)

#endif /* PPB_MARKER_ENABLED */
