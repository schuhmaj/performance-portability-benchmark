#pragma once

#include "benchmark/benchmark.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>

namespace ppb {
    /**
     * Simple Class offering a vector addition benchmark
     * @tparam FloatType
     */
    template<typename FloatType>
    class VectorAddition {

        /** The size of the vector addition */
        size_t _size;

        /** First input vector to be summed */
        std::vector<FloatType> _inA;

        /** Second input vector to be summed */
        std::vector<FloatType> _inB;

        struct impl;
        std::unique_ptr<impl> _impl{nullptr};

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
            this->init();
        }

        /** Default Destructor */
        ~VectorAddition() = default;

        /**
         * Initializes the internal implementation of the class (the _impl member).
         * This method is called in the constructor of the class.
         */
        void init();

        /**
         * Performs the vector addition and returns the _outC vector.
         * @return results of inA + inB
         */
        std::vector<FloatType> operator()();

        /**
         * Static method for benchmarking the vector addition, i.e. the operator() of the VectorAddition class
         * The time measurements are conducted using std::chrono::high_resolution_clock.
         * The complexity, i.e. the size N of the vectors, is inserted into the Google Benchmark state object along
         * the time measurements.
         * @param state the benchmarking state, must contain a range determining the size of the vectors to be added
         */
        static void inline benchmark(benchmark::State& state) {
            const size_t size = state.range(0);
            VectorAddition<FloatType> vec{size};

            for (auto _ : state) {
                const auto start = std::chrono::high_resolution_clock::now();

                auto result = vec();
                benchmark::DoNotOptimize(result);

                const auto end = std::chrono::high_resolution_clock::now();
                const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                state.SetIterationTime(elapsed_seconds.count());

                // Sanity Check that the Vector Addition was actually successful (this is not in the benchmark by design)
                if (result[1] != vec._inA[1] + vec._inB[1]) {
                    std::stringstream ss;
                    ss << "Vector addition failed! " << result[1] << " != " << vec._inA[1] << " + " << vec._inB[1];
                    throw std::runtime_error(ss.str());
                }
            }
            state.SetComplexityN(static_cast<long long>(size));
        }

    };
} // namespace ppb
