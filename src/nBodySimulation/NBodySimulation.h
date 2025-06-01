
#pragma once
#include "benchmark/benchmark.h"

namespace ppb {

    template<typename FloatType>
    class NBodySimulation {



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
} // namespace ppb