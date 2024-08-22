#pragma once

#include "benchmark/benchmark.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

template<typename FloatType>
class VectorAddition {

private:

    std::vector<FloatType> _inA;
    std::vector<FloatType> _inB;
    std::vector<FloatType> _outC;

public:

    explicit VectorAddition(size_t size) : _inA(size), _inB(size), _outC(size) {
        std::iota(_inA.begin(), _inA.end(), 0);
        std::iota(_inB.begin(), _inB.end(), 0);
        std::fill(_outC.begin(), _outC.end(), 0);
    }

    ~VectorAddition() = default;

    std::vector<FloatType> operator()();

    static void inline vectorAdditionBenchmark(benchmark::State& state) {
        const size_t size = state.range(0);
        VectorAddition<FloatType> vec{size};

        for (auto _ : state) {
            const auto start = std::chrono::high_resolution_clock::now();

            auto result = vec();
            benchmark::DoNotOptimize(result);

            const auto end = std::chrono::high_resolution_clock::now();
            const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            state.SetIterationTime(elapsed_seconds.count());
        }
        state.SetComplexityN(static_cast<long long>(size));
    }

};


BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)->Name("Vector Addition Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();
BENCHMARK(VectorAddition<double>::vectorAdditionBenchmark)->Name("Vector Addtion Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();
