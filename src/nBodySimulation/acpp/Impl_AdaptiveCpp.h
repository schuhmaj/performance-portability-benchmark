/**
 * @file Impl_AdaptiveCpp.h
 *
 * Implements a classical n-body simulation using the Lennard-Jones potential and a simple velocity Verlet integrator.
 * Provides methods for updating particle positions, velocities, and computing inter-particle forces according
 * to the given simulation configuration. Designed to be used as a backend for benchmarking or analysis.
 */

#pragma once

#include <sycl/sycl.hpp>

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "ParticleContainer.h"
#include "common/UtilityContainer.h"

namespace ppb {

    /**
     * @class ImplAdaptiveCpp
     * Template for a classical n-body simulation using the Lennard-Jones potential and velocity Verlet integration.
     *
     * @tparam FloatType Floating-point type for simulation (float or double).
     */
    template<typename FloatType>
    class ImplAdaptiveCpp {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Stores tje timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

        /**
         * Manages the tasks on a device
         */
        sycl::queue _queue;

        /**
         * Particle Data in SoA form on device
         */
        sycl::vec<FloatType, 4> *_positions = nullptr;
        sycl::vec<FloatType, 4> *_velocities = nullptr;
        sycl::vec<FloatType, 4> *_forces = nullptr;
        sycl::vec<FloatType, 4> *_oldForces = nullptr;

        size_t _size = 0;

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
         * Runs the simulation for the configured total simulation time, performing position, force,
         * and velocity updates for each step using the velocity Verlet scheme.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::vector<Particle<FloatType>> Final state of all particles after the simulation.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

    private:

        /**
         * Updates positions of all particles using velocity Verlet integration and resets their forces
         * with the configured global force.
         *
         * @param queue Sycl device queue, accessor of execution target
         * @param particlesUSM Array of particles in shared memory whose positions and forces are modified in-place.
         * @param size size of particleusm, amount of particles
         */
        void updatePositionsAndResetForce();

        /**
         * Updates velocities of all particles using the forces computed before and after the integration step.
         *
         * @param queue Sycl device queue, accessor of execution target
         * @param particlesUSM Array of particles in shared memory whose positions and forces are modified in-place.
         * @param particles Vector of particles for which the forces will be calculated and accumulated.
         */
        void updateVelocities();

        /**
         * Computes the inter-particle forces for all particles using the Lennard-Jones potential.
         *
         * @param queue Sycl device queue, accessor of execution target
         * @param particlesUSM Array of particles in shared memory whose positions and forces are modified in-place.
         * @param particles Vector of particles for which the forces will be calculated and accumulated.
         */
        void computeForces_atomic();

        /**
         * Computes the inter-particle forces for all particles of neighboring cells using the Lennard-Jones potential.
         *
         * @param queue Sycl device queue, accessor of execution target
         * @param particlesUSM Array of particles in shared memory whose positions and forces are modified in-place.
         * @param particles Vector of particles for which the forces will be calculated and accumulated.
         */
        void computeForces_cell_based_atomic(ParticleContainer<FloatType> *particle_container);

    };
} // namespace ppb