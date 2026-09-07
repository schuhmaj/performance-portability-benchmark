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
 *  - the benchmark configuration headers collapse their size range to the
 *    single largest input (see MIN_SIZE/MAX_SIZE in VectorAddition.h,
 *    MatrixMultiplication.h and NBodySimulation.h, and the registrations in
 *    polyhedralGravity/main.cpp), and
 *  - ppb::profiling::initialize() below pins Google Benchmark to a single
 *    iteration and a single repetition.
 *
 * The resulting binary needs no extra command-line arguments to be profiled,
 * which is what the `ppbcc profile` batch driver relies on.
 */

#include <benchmark/benchmark.h>

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
