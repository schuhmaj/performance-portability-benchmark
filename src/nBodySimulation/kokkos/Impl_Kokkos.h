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
#include "NBodySimulation.h"
#include "Particle.h"
#include "UtilityContainer.h"

namespace ppb {

    /**
     * @class ImplKokkos
     * Templated n-body simulation using Kokkos for parallelism, the Lennard-Jones potential, and velocity Verlet integration.
     *
     * @tparam FloatType Floating-point type for simulation (e.g., float or double).
     */
    template<typename FloatType>
    class ImplKokkos {

        /**
         * Simulation configuration which holds parameters such as particle count, global forces, simulation time, etc.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Type alias for a Kokkos View of Particle<FloatType> on the device.
         */
        using ParticleView = Kokkos::View<Particle<FloatType>*>;
        ParticleView particlesDevice;

        /**
         * Host mirror for the device particle view, used for data transfers between host and device.
         */
        typename ParticleView::HostMirror particlesHost;

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

        /**
         * Runs the simulation for the configured total time using parallel Kokkos kernels to update
         * positions, velocities, and compute forces at each step.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::vector<Particle<FloatType>> Final state of all particles after the simulation.
         */
        std::vector<Particle<FloatType>> simulate(const std::vector<Particle<FloatType>> &particles);

    private:

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