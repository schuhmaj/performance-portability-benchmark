#pragma once
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include <iostream>
#include <optional>
#include <chrono>
#include <openacc.h>

namespace ppb {

    template <typename FloatType>
    class OpenACCParticleSoA {

        /**
         * Reference to the original vector of particles (used as a data source during initialization and conversion).
         */
        const std::vector<Particle<FloatType>> &_ref;

    public:

        // Device-resident raw pointers (OpenACC-managed)
        FloatType* positions{nullptr};
        FloatType* velocities{nullptr};
        FloatType* forces{nullptr};
        FloatType* oldForces{nullptr};

        // Host mirrors
        std::vector<FloatType> positionsHost;
        std::vector<FloatType> velocitiesHost;
        std::vector<FloatType> forcesHost;

        explicit OpenACCParticleSoA(const std::vector<Particle<FloatType>> &ref);

        ~OpenACCParticleSoA();

        std::vector<Particle<FloatType>> toParticles();

    };

    template<typename FloatType>
    class ImplOpenACC {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

        std::optional<OpenACCParticleSoA<FloatType>> _particles{std::nullopt};

    public:

        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplOpenACC(const ParticleSimulationConfig<FloatType> &config);

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
         */
        void updatePositionsAndResetForce();

        /**
         * Updates velocities of all particles using the forces computed before and after the integration step.
         */
        void updateVelocities();

        /**
         * Computes the inter-particle forces for all particles using the Lennard-Jones potential.
         */
        void computeForces();


    };
} // namespace ppb