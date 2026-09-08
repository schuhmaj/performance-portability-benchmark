#pragma once

/**
 * @file Profiling.h
 *
 * Support for the CMake option PPB_PROFILING.
 *
 * A benchmark executable is built for statistics: it sweeps a whole range of
 * problem sizes and repeats every size until Google Benchmark is happy with the
 * noise level. A kernel profiler such as Nvidia Nsight Compute (ncu) wants the
 * exact opposite - one problem size, replayed exactly once - because it
 * serializes and replays every single kernel launch it sees.
 *
 * PPB_PROFILING switches the executables into that second mode:
 *  - the benchmark configuration headers collapse their size range to a single
 *    input (see MIN_SIZE/MAX_SIZE in VectorAddition.h, MatrixMultiplication.h
 *    and NBodySimulation.h, and the registrations in
 *    polyhedralGravity/main.cpp). It is the largest input except where that
 *    would defeat the profiler: matrix multiplication profiles 4096 rather
 *    than 16384 because a replay pass has to snapshot several GB at the
 *    larger size, and the polyhedral benchmark profiles the largest mesh
 *    rather than Eros because Eros' kernels are only a few microseconds long,
 *  - ppb::profiling::initialize() below pins Google Benchmark to a single
 *    iteration and a single repetition,
 *  - PPB_ENABLE_NVTX is switched on, so the regions of common/Marker.h name
 *    the kernels they launch ("matmul", "init", "evaluate"). A profiler names a
 *    kernel whatever the compiler called it, which for several paradigms is
 *    nothing useful - AdaptiveCpp launches four kernels all called
 *    __acpp_sscp_kernel - and this is what tells them apart, and
 *  - the targets are compiled with line tables (see src/CMakeLists.txt), which
 *    lets a profiler map a hot instruction back to a source line. The
 *    optimization flags are untouched on purpose: a roofline measured from a
 *    differently optimized binary describes a different program.
 *
 * The resulting binary needs no extra command-line arguments to be profiled,
 * which is what the `ppbcc profile` batch driver relies on.
 */

#include <benchmark/benchmark.h>

#include "common/Marker.h"

#ifdef PPB_ENABLE_LIKWID
#include <cstdlib>
#endif

#ifdef PPB_PROFILING
#include <vector>
#endif

namespace ppb::profiling {

#ifdef PPB_PROFILING
    /** Whether this executable was built in profiling mode. */
    inline constexpr bool enabled = true;
#else
    inline constexpr bool enabled = false;
#endif

    /**
     * Drop-in replacement for benchmark::Initialize.
     *
     * Without PPB_PROFILING the call is forwarded verbatim. With PPB_PROFILING
     * the flags pinning the run to one un-repeated iteration are prepended to
     * the argument list. They are prepended (not appended) so an explicit flag
     * on the command line still wins: Google Benchmark keeps the last
     * occurrence of a flag.
     *
     * @param argc pointer to main's argument count
     * @param argv main's argument vector
     */
    inline void initialize(int *argc, char **argv) {
#ifdef PPB_ENABLE_LIKWID
        // The marker API has to be opened before the first region and closed
        // after the last one. Every main() funnels through here, and atexit
        // gives the matching close without touching any of them again.
        PPB_MARKER_INIT;
        std::atexit([]() { PPB_MARKER_CLOSE; });
#endif
#ifdef PPB_PROFILING
        static char minTime[] = "--benchmark_min_time=1x";
        static char repetitions[] = "--benchmark_repetitions=1";

        static std::vector<char *> arguments;
        arguments.clear();
        arguments.push_back(argv[0]);
        arguments.push_back(minTime);
        arguments.push_back(repetitions);
        arguments.insert(arguments.end(), argv + 1, argv + *argc);
        int count = static_cast<int>(arguments.size());
        arguments.push_back(nullptr);

        benchmark::Initialize(&count, arguments.data());
#else
        benchmark::Initialize(argc, argv);
#endif
    }

}// namespace ppb::profiling
