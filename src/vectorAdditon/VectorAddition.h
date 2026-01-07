#pragma once

#include "benchmark/benchmark.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <utility>
#include <exception>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <sys/stat.h>

namespace ppb {

    namespace VectorAdditionBenchmarkConf {
        constexpr double MIN_SIZE = 1e3;
        constexpr double MAX_SIZE = 1e8;
    }

    /**
     * Simple Class offering a vector addition benchmark
     */
    template<class Implementation>
    class VectorAddition {

        using FloatType = typename Implementation::float_type;

        /** The size of the vector addition */
        size_t _size;

        /** First input vector to be summed */
        std::vector<FloatType> _inA;

        /** Second input vector to be summed */
        std::vector<FloatType> _inB;

        Implementation _impl;

    public:
        /**
         * Constructs a new Vector Addition class with a given size.
         * Initializes the classes' input vectors with incrementing numbers starting from zero and
         * fully zeros the output vector.
         * @param size - the size of the vector addition
         */
        explicit VectorAddition(const size_t size) : _size{size}, _inA(size), _inB(size) {
            std::iota(_inA.begin(), _inA.end(), 0);
            std::iota(_inB.begin(), _inB.end(), 0);
        }

        /** Default Destructor */
        ~VectorAddition() = default;

        /**
         * Performs the vector addition and returns the _outC vector.
         * @return results of inA + inB
         */
        std::pair<std::vector<FloatType>, double> operator()() {
            return _impl(_inA, _inB);
        }

        /**
         * Static method for benchmarking the vector addition, i.e. the operator() of the VectorAddition class
         * The time measurements are conducted using std::chrono::high_resolution_clock.
         * The complexity, i.e. the size N of the vectors, is inserted into the Google Benchmark state object along
         * the time measurements.
         * @param state the benchmarking state, must contain a range determining the size of the vectors to be added
         */
        static void inline benchmark(benchmark::State& state) {
            const size_t size = state.range(0);
            VectorAddition vec{size};

#ifndef PPB_MEASURE_ONLY_KERNEL
            double kernelTime = 0;
#endif
            for (auto _ : state) {
                auto [result, time] = vec();
                benchmark::DoNotOptimize(result);
                // This time is only used when UseManualTime() is enabled for the
                state.SetIterationTime(time);
#ifndef PPB_MEASURE_ONLY_KERNEL
                kernelTime += time;
#endif

                // Sanity Check that the Vector Addition was actually successful (this is not in the benchmark by design)
                if (result[1] != vec._inA[1] + vec._inB[1]) {
                    std::stringstream ss{};
                    ss << "Vector addition failed! " << result[1] << " != " << vec._inA[1] << " + " << vec._inB[1];
                    throw std::runtime_error(ss.str());
                }
            }
#ifndef PPB_MEASURE_ONLY_KERNEL
            state.counters["kernel_time"] = benchmark::Counter(kernelTime, benchmark::Counter::kAvgIterations);
#endif
            state.SetComplexityN(static_cast<long long>(size));
        }

    };
} // namespace ppb
