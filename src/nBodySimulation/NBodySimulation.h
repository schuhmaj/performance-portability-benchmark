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
#include <utility>

#include "Particle.h"
#include "UtilityContainer.h"
#include "benchmark/benchmark.h"

namespace ppb {

    namespace NBodyBenchmarkConf {
        constexpr double MIN_SIZE = 1e6;
        constexpr double MAX_SIZE = 1e7;
    }

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
        int numberTimeSteps{1000};

        /**
         * Size of the simulation time step.
         */
        FloatType deltaT{1e-10};

        /**
         * Global force applied to all particles (3D vector).
         */
        std::array<FloatType, 3> globalForce{0, 0, 0};

        /**
         * Minimum box coordinates of the initial simulation domain (lower-left-corner)
         */
        std::array<FloatType, 3> boxMin{-10, -10, -10};

        /**
         * Maximum box coordinates of the initial simulation domain (upper-right-corner)
         */
        std::array<FloatType, 3> boxMax{10, 10, 10};

        /**
         * Seed to initialize the ParticleGenerator
         */
        unsigned int seed{42};

        /**
         * Cell size used in the linked cell implementation (cell_size >= cutoff_radius!!!)
         */
        FloatType cell_size{20.0f};
        
        /**
         * Cutoff radius used in the linked cell and verlet lists implementation
         */
        FloatType cutoff_radius{20.0f};

        /**
        * Size of the Verlet skin.
        */
        FloatType verlet_skin{2.0f};

        /**
        * Frequency of updates of verlet lists. 
        * Updates the verlet lists every 'frequency' iterations.
        */
        size_t frequency{15};

        /**
         * Creates a simulation configuration.
         * @param size Number of particles in the simulation.
         */
        explicit ParticleSimulationConfig(const size_t size) : size{size} {}

        ParticleSimulationConfig(const size_t size, const int numberTimeSteps, const FloatType deltaT)
            : size{size}
            , numberTimeSteps{numberTimeSteps}
            , deltaT{deltaT}
        {}
    };


    struct ParticleSimulationTimings {
        /** Total accumulated time for position updates and force reset in nanoseconds [ns] */
        double positionUpdateForceResetTime;
        /** Total accumulated time for velocity updates in nanoseconds [ns] */
        double velocityUpdateTime;
        /** Total accumulated time for force updates in nanoseconds [ns] */
        double forceUpdateTime;

        ParticleSimulationTimings operator+(const ParticleSimulationTimings &other) const {
            return {positionUpdateForceResetTime + other.positionUpdateForceResetTime, velocityUpdateTime + other.velocityUpdateTime, forceUpdateTime + other.forceUpdateTime};
        }
        ParticleSimulationTimings operator+=(const ParticleSimulationTimings &other) {
            positionUpdateForceResetTime += other.positionUpdateForceResetTime;
            velocityUpdateTime += other.velocityUpdateTime;
            forceUpdateTime += other.forceUpdateTime;
            return *this;
        }

        void reset() {
            positionUpdateForceResetTime = 0.0;
            velocityUpdateTime = 0.0;
            forceUpdateTime = 0.0;
        }
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
        {impl.simulate(particles) } -> std::convertible_to<std::pair<std::vector<Particle<typename Impl::float_type>>, ParticleSimulationTimings>>;
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
            : _particles{Particle<FloatType>::generateUniform(config.boxMin, config.boxMax, config.size, config.seed)}
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
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> operator()() {
            return _impl.simulate(_particles);
        }

        /**
         * Method suitable for Google Benchmark framework to measure performance.
         * @param state Benchmark state.
         */
        static void inline benchmark(benchmark::State& state) {
            const size_t size = state.range(0);
            NBodySimulation nbodySimulation{size};

            ParticleSimulationTimings totalTimings{};
            for (auto _ : state) {
                auto [result, iterationTimings] = nbodySimulation();
                benchmark::DoNotOptimize(result);
                totalTimings += iterationTimings;
            }
            const auto&[positionUpdateForceResetTime, velocityUpdateTime, forceUpdateTime] = totalTimings;
            state.counters["position_update_reset"] = benchmark::Counter(positionUpdateForceResetTime, benchmark::Counter::kAvgIterations);
            state.counters["velocity_update"] = benchmark::Counter(velocityUpdateTime, benchmark::Counter::kAvgIterations);
            state.counters["force_update"] = benchmark::Counter(forceUpdateTime, benchmark::Counter::kAvgIterations);
            state.SetComplexityN(static_cast<long long>(size));
        }

    };
} // namespace ppb