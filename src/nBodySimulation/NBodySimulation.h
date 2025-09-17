/**
 * @file NBodySimulation.h
 *
 * Contains classes, concepts, and configuration structures that provide a flexible and portable benchmark framework
 * for n-body particle simulations. The main components allow for plugging in different implementation strategies,
 * using a configuration to set up simulation properties such as particle count, time step, and bounding box.
 * The framework integrates with Google Benchmark for performance measurement.
 *
 * - ParticleSimulationConfig: Holds configuration for an n-body simulation.
 * - ParticleSimulationImpl: Concept that checks if an implementation provides required types and methods.
 * - NBodySimulation: A generic benchmarking wrapper for simulation implementations that satisfy the concept.
 *
 * All types and logic are contained within the namespace ppb.
 */

#pragma once
#include <chrono>
#include <memory>

#include "Particle.h"
#include "UtilityContainer.h"
#include "benchmark/benchmark.h"

namespace ppb {

    /**
     * Structure defining the configuration for an n-body particle simulation.
     *
     * @tparam FloatType Floating point type used for computation (e.g., float, double).
     */
    template<typename FloatType>
    struct ParticleSimulationConfig {
        /**
         * Number of particles in the simulation.
         */
        size_t size;

        /**
         * Simulation end time.
         */
        FloatType numberTimeSteps{1000};

        /**
         * Size of the simulation time step.
         */
        FloatType deltaT{1e-10};

        /**
         * Global force applied to all particles (3D vector).
         */
        std::array<FloatType, 3> globalForce{0, 0, 0};

        /**
         * Minimum coordinates of the simulation bounding box.
         */
        std::array<FloatType, 3> boxMin{0, 0, 0};

        /**
         * Maximum coordinates of the simulation bounding box.
         */
        std::array<FloatType, 3> boxMax{1, 1, 1};

        /**
         * Creates a simulation configuration.
         * @param size Number of particles in the simulation.
         */
        explicit ParticleSimulationConfig(const size_t size) : size{size} {}
    };

    /**
     * Concept that validates a type Impl as a valid n-body simulation implementation.
     *
     * Requirements:
     * - Impl must provide a nested type float_type.
     * - Impl must be constructible from ParticleSimulationConfig<float_type>.
     * - Impl must provide a simulate() method taking a vector of Particle<float_type> and returning
     *   a vector of Particle<float_type>.
     *
     * Usage: Used to constrain templates to valid implementation types for the simulation framework.
     */
#if __cplusplus >= 202002L
    template <typename Impl>
    concept ParticleSimulationImpl = requires(Impl impl, ParticleSimulationConfig<typename Impl::float_type> config, const std::vector<Particle<typename Impl::float_type>> &particles) {
        typename Impl::float_type;
        {Impl {config}};
        {impl.simulate(particles) } -> std::convertible_to<std::vector<Particle<typename Impl::float_type>>>;
    };

    /**
     * Generic benchmarking wrapper for an n-body simulation algorithm implementation.
     *
     * @tparam ParticleSimulationImpl The simulation implementation to use (must satisfy the ParticleSimulationImpl concept).
     */
    template<ParticleSimulationImpl ParticleSimulationImpl>
#else
    template<class ParticleSimulationImpl>
#endif
    class NBodySimulation {

        /**
         * The floating point type used by the simulation (extracted from the implementation).
         */
        using FloatType = typename ParticleSimulationImpl::float_type;

        /**
         * Vector holding the simulation's particles.
         */
        std::vector<Particle<FloatType>> _particles;

        /**
         * The simulation implementation instance.
         */
        ParticleSimulationImpl _impl;

    public:

        /**
         * Constructs the simulation, generating an initial cuboid set of particles, and initializing the implementation.
         *
         * @param config The ParticleSimulationConfig<FloatType> holding setup for particle number, box bounds, force, etc.
         */
        explicit NBodySimulation(const ParticleSimulationConfig<FloatType> &config)
            : _particles{Particle<FloatType>::generateCuboid(config.boxMin, config.boxMax, config.size)}
            , _impl{config}
        {}

        /**
         * Constructs the simulation, generating an initial cuboid set of particles, and initializing the implementation.
         *
         * @param size the number of particles in the cuboid
         */
        explicit NBodySimulation(const size_t size) : NBodySimulation(ParticleSimulationConfig<FloatType>(size)) {}

        /**
         * Executes the simulation and returns the resulting particles.
         *
         * @return the particle vector resulting from the implementation's simulate() call.
         */
        std::vector<Particle<FloatType>> operator()() {
            return _impl.simulate(_particles);
        }

        /**
         * Method suitable for Google Benchmark framework to measure performance.
         * @param state Benchmark state.
         */
        static void inline benchmark(benchmark::State& state) {
            const size_t size = state.range(0);
            NBodySimulation nbodySimulation{size};

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