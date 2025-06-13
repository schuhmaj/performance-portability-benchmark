#pragma once
#include <chrono>
#include <memory>

#include "Particle.h"
#include "benchmark/benchmark.h"

namespace ppb {

    template<typename FloatType>
    class NBodySimulation {

        using ParticleContainer = std::vector<Particle<FloatType>>;

        ParticleContainer _particles;

        double _endT;

        double _deltaT;

        std::array<FloatType, 3> _globalForce{};

        struct impl;
        std::unique_ptr<impl> _impl{nullptr};

        void init();

    public:

        explicit NBodySimulation(size_t size, double endT = 1, double deltaT = 0.001) : _particles{Particle<FloatType>::generateCuboid({0,0,0}, {1, 1, 1}, size)}, _endT{endT}, _deltaT {deltaT} {
            this->init();
        }

        ParticleContainer operator()();

        std::vector<Particle<FloatType>> getParticles() {
            return _particles;
        }

        static void inline benchmark(benchmark::State& state) {
            const size_t size = state.range(0);
            NBodySimulation<FloatType> nbodySimulation{size};

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