/**
 * @file Impl_Cpp.h
 *
 * Implements a classical n-body simulation using the Lennard-Jones potential and a simple velocity Verlet integrator.
 * Provides methods for updating particle positions, velocities, and computing inter-particle forces according
 * to the given simulation configuration. Designed to be used as a backend for benchmarking or analysis.
 */

#pragma once
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include <iostream>
#include <chrono>

// #define PPB_NBODY_ENABLE_CUTOFF

namespace ppb {

    /**
     * @def PPB_NBODY_ENABLE_CUTOFF
     * Debug switch for the Lennard-Jones cut-off radius of ImplCpp.
     *
     * It is deliberately NOT wired into the CMake system yet; compile the nbody_cpp_lib sources with
     * -DPPB_NBODY_ENABLE_CUTOFF to enable it. When enabled, pair interactions beyond the cut-off radius are skipped
     * and every call of computeForces() reports to stdout how many pairs were skipped and how many were evaluated.
     *
     * NOTE: Switching this on changes the trajectory and therefore breaks the reference comparison of NBodyTest,
     * which was generated with an effectively infinite cut-off.
     */

    /**
     * @class ImplCpp
     * Template for a classical n-body simulation using the Lennard-Jones potential and velocity Verlet integration.
     *
     * @tparam FloatType Floating-point type for simulation (float or double).
     */
    template<typename FloatType>
    class ImplCpp {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

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
         * Factor determining the cut-off radius from the (mixed) Lennard-Jones sigma of an interacting pair:
         * r_c = CUTOFF_SIGMA_FACTOR * sigma_ij.
         *
         * 2.5 * sigma is the textbook choice for the Lennard-Jones potential and is derived from sigma alone:
         * epsilon only scales the potential, it does not move the point at which the potential becomes negligible.
         * At r = 2.5 * sigma the potential has decayed to |U| ~ 0.0163 * epsilon (1.6% of the well depth) and the
         * magnitude of the force to ~0.4% of the strongest attractive force, so truncating there is safe.
         * With the sigma = 1 / epsilon = 1 particles of this benchmark this yields r_c = 2.5.
         */
        static constexpr FloatType CUTOFF_SIGMA_FACTOR{2.5};

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplCpp(const ParticleSimulationConfig<FloatType> &config);

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
         * @param particles Vector of particles whose positions and forces are modified in-place.
         */
        void updatePositionsAndResetForce(std::vector<Particle<FloatType>> &particles);

        /**
         * Updates velocities of all particles using the forces computed before and after the integration step.
         *
         * @param particles Vector of particles whose velocities are modified in-place.
         */
        void updateVelocities(std::vector<Particle<FloatType>> &particles);

        /**
         * Computes the inter-particle forces for all particles using the Lennard-Jones potential.
         *
         * @param particles Vector of particles for which the forces will be calculated and accumulated.
         */
        void computeForces(std::vector<Particle<FloatType>> &particles);


    };
} // namespace ppb