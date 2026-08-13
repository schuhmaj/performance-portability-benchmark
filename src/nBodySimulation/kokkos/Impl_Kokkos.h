/**
 * @file Impl_Kokkos.h
 *
 * Implements an n-body simulation using Kokkos for portability across various parallel architectures.
 * Utilizes the Lennard-Jones potential and a velocity Verlet integrator for the motion and force calculations.
 * This implementation enables high-performance execution on CPUs and GPUs by leveraging the Kokkos abstraction layer.
 * Designed to be used as a backend for benchmarking and performance comparison.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {

    /**
     * @struct KokkosParticleSoA
     * Structure of Arrays (SoA) container for particle attributes optimized for Kokkos parallelism.
     *
     * Stores particle positions, velocities, forces, and related data in device views for efficient
     * bulk memory access and parallel computation on CPUs or GPUs using Kokkos. The data is laid out
     * in a structure-of-arrays style, enabling high memory throughput and coalesced access patterns.
     *
     * Host mirror views are provided for transferring data between host and device memory spaces.
     *
     * @tparam FloatType Floating-point type for numeric data (e.g., float or double).
     */
    template <typename FloatType>
    struct KokkosParticleSoA {
        /**
         * Device view of particle positions, shape [N][3].
         */
        Kokkos::View<FloatType *[3]> positions;

        /**
         * Host mirror view of particle positions, shape [N][3].
         */
        typename Kokkos::View<FloatType *[3]>::host_mirror_type positionsHost;

        /**
         * Device view of particle velocities, shape [N][3].
         */
        Kokkos::View<FloatType *[3]> velocities;

        /**
         * Host mirror view of particle velocities, shape [N][3].
         */
        typename Kokkos::View<FloatType *[3]>::host_mirror_type velocitiesHost;

        /**
         * Device view of particle forces, shape [N][3].
         */
        Kokkos::View<FloatType *[3]> forces;

        /**
         * Host mirror view of particle forces, shape [N][3].
         */
        typename Kokkos::View<FloatType *[3]>::host_mirror_type forcesHost;

        /**
         * Device view of previous forces for velocity Verlet integration, shape [N][3].
         */
        Kokkos::View<FloatType *[3]> oldForces;

        /**
         * Reference to the original vector of particles (used as a data source during initialization and conversion).
         */
        const std::vector<Particle<FloatType>> &_ref;

        /**
         * Constructs a KokkosParticleSoA from a standard vector of Particle objects.
         * Allocates device and host mirror memory and copies the input particle data
         * into the SoA structure for efficient parallel access.
         *
         * @param particles Input vector of Particle<FloatType> objects to initialize from.
         */
        explicit KokkosParticleSoA(const std::vector<Particle<FloatType>> &particles);

        /**
         * Copies the current SoA data back into a standard vector of Particle objects.
         * This involves copying data from device to host and assembling Particle instances.
         *
         * @return A vector of Particle<FloatType> reflecting the state stored in device views.
         */
        std::vector<Particle<FloatType>> toParticles();

        /**
         * Returns the number of particles managed by this structure.
         *
         * @return Particle count (size of the arrays).
         */
        size_t size() const;
    };

    /**
     * @class ImplKokkos
     * Templated n-body simulation using Kokkos for parallelism, the Lennard-Jones potential, and velocity Verlet
     * integration.
     *
     * @tparam FloatType Floating-point type for simulation (e.g., float or double).
     */
    template <typename FloatType>
    class ImplKokkos {
    protected:

        /**
         * Simulation configuration which holds parameters such as particle count, global forces, simulation time, etc.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * The SoA GPU structure. It is initialized each time the simulate() functions is called
         */
        std::optional<KokkosParticleSoA<FloatType>> _particles{std::nullopt};

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

    public:
        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplKokkos(const ParticleSimulationConfig<FloatType> &config);

        virtual ~ImplKokkos() = default;

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
        virtual void computeForces();
    };
} // namespace ppb
