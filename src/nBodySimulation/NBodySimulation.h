#pragma once
#include "benchmark/benchmark.h"

namespace ppb {

    template<typename Container>
    class NBodySimulation {

        Container _particles;


        Container operator()();


        static void inline benchmark(benchmark::State& state) {
            const size_t size = state.range(0);
            NBodySimulation<Container> nbodySimulation{size};

            for (auto _ : state) {
                const auto start = std::chrono::high_resolution_clock::now();

                auto result = nbodySimulation();
                benchmark::DoNotOptimize(result);

                const auto end = std::chrono::high_resolution_clock::now();
                const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                state.SetIterationTime(elapsed_seconds.count());
            }
            state.SetComplexityN(static_cast<long long>(size));
        }


    };
} // namespace ppb