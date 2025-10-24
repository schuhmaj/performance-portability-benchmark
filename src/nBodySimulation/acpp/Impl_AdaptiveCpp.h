#pragma once

#include <sycl/sycl.hpp>
#include <optional>
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {


    template <typename FloatType>
    struct AdaptiveCppParticleSoA {

        static constexpr size_t ALIGNMENT = 64;
        sycl::queue &_queue;

        FloatType *positions;
        FloatType *velocities;
        FloatType *forces;
        FloatType *oldForces;


        const std::vector<Particle<FloatType>> &_ref;

        explicit AdaptiveCppParticleSoA(const std::vector<Particle<FloatType>> &particles, sycl::queue &queue);

        std::vector<Particle<FloatType>> toParticles();
    };

    /**
     * @class ImplKokkos
     * Templated n-body simulation using Kokkos for parallelism, the Lennard-Jones potential, and velocity Verlet
     * integration.
     *
     * @tparam FloatType Floating-point type for simulation (e.g., float or double).
     */
    template <typename FloatType>
    class ImplAdaptiveCpp {

        /**
         * Simulation configuration which holds parameters such as particle count, global forces, simulation time, etc.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * The SoA GPU structure. It is initialized each time the simulate() functions is called
         */
        std::optional<AdaptiveCppParticleSoA<FloatType>> _particles{std::nullopt};

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

        sycl::queue _queue;

    public:
        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplAdaptiveCpp(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Runs the simulation for the configured total time using parallel Kokkos kernels to update
         * positions, velocities, and compute forces at each step.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::vector<Particle<FloatType>> Final state of all particles after the simulation.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

        /**
         * Updates positions of all particles on the device using the velocity Verlet integrator,
         * and resets each particle's force to the configured global force in parallel.
         */
        void updatePositionsAndResetForce();

        /**
         * Updates velocities of all particles on the device based on forces before and after the integration
         * step, using parallel execution.
         */
        void updateVelocities();

        /**
         * Computes the inter-particle forces using the Lennard-Jones potential for all particles on the device,
         * accumulating the results in parallel.
         */
        void computeForces();
    };
} // namespace ppb
