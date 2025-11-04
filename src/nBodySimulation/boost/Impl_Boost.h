#pragma once
#include "boost/compute.hpp"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "ForceKernel.h"
#include "common/UtilityContainer.h"
#include <iostream>
#include <chrono>
#include <optional>

namespace ppb {

    template <typename FloatType>
    class BoostParticleSoA {

        /**
         * Reference to the original vector of particles (used as a data source during initialization and conversion).
         */
        const std::vector<Particle<FloatType>> &_ref;

    public:

        boost::compute::vector<boost::compute::float4_> positions;
        boost::compute::vector<boost::compute::float4_> velocities;
        boost::compute::vector<boost::compute::float4_> forces;
        boost::compute::vector<boost::compute::float4_> oldForces;

        std::vector<boost::compute::float4_> positionsHost;
        std::vector<boost::compute::float4_> velocitiesHost;
        std::vector<boost::compute::float4_> forcesHost;

        boost::compute::command_queue &queue;


        BoostParticleSoA(const std::vector<Particle<FloatType>> &ref, const boost::compute::context &context, boost::compute::command_queue &queue);

        std::vector<Particle<FloatType>> toParticles();

    };

    /**
     * @class ImplBoost
     * Template for a classical n-body simulation using the Lennard-Jones potential and velocity Verlet integration.
     *
     * @tparam FloatType Floating-point type for simulation (float or double).
     */
    template<typename FloatType>
    class ImplBoost {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

        /**
         * The SoA GPU structure. It is initialized each time the simulate() functions is called
         */
        std::optional<BoostParticleSoA<FloatType>> _particles{std::nullopt};
        const unsigned int _numParticles;
        const float _deltaT;
        const boost::compute::float4_ _globalForce;

        boost::compute::device gpu;
        boost::compute::context context;
        boost::compute::command_queue queue;
        boost::compute::program program;
        boost::compute::kernel kernelPositionUpdate;
        boost::compute::kernel kerneVelocityUpdate;
        boost::compute::kernel kernelForceUpdate;

    public:

        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplBoost(const ParticleSimulationConfig<FloatType> &config);

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
         * Sets up the GPU datastructures and the Kernel Arguments.
         * This is called once before the simulation starts by simulate()
         * @param particles the particles to be transfered to the GPU-SoA
         */
        void init(const std::vector<Particle<FloatType>> &particles);

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