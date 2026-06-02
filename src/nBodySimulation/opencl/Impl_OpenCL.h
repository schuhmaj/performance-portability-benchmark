#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "common/opencl/OpenCLUtility.h"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "ForceKernel.h"
#include "common/UtilityContainer.h"
#include <iostream>
#include <chrono>
#include <optional>

namespace ppb {

    template <typename FloatType> struct cl_vec_traits;
    template <> struct cl_vec_traits<float>  { using scalar = cl_float;  using vec4 = cl_float4;  };
    template <> struct cl_vec_traits<double> { using scalar = cl_double; using vec4 = cl_double4; };

    template <typename FloatType>
    class OpenCLParticleSoA {

        using cl_scalar = typename cl_vec_traits<FloatType>::scalar;
        using cl_vec4 = typename cl_vec_traits<FloatType>::vec4;

        /**
         * Reference to the original vector of particles (used as a data source during initialization and conversion).
         */
        const std::vector<Particle<FloatType>> &_ref;

    public:

        cl_mem positions;
        cl_mem velocities;
        cl_mem forces;
        cl_mem oldForces;

        std::vector<cl_vec4> positionsHost;
        std::vector<cl_vec4> velocitiesHost;
        std::vector<cl_vec4> forcesHost;

        OpenCLParticleSoA(const std::vector<Particle<FloatType>> &ref, cl_context &context);

        ~OpenCLParticleSoA();

        std::vector<Particle<FloatType>> toParticles(cl_command_queue &queue);

    };

    /**
     * @class ImplOpenCL
     * Template for a classical n-body simulation using the Lennard-Jones potential and velocity Verlet integration.
     *
     * @tparam FloatType Floating-point type for simulation (float or double).
     */
    template<typename FloatType>
    class ImplOpenCL {

        using cl_scalar = typename cl_vec_traits<FloatType>::scalar;
        using cl_vec4 = typename cl_vec_traits<FloatType>::vec4;

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings{};

        /**
         * The SoA GPU structure. It is initialized each time the simulate() functions is called
         */
        std::optional<OpenCLParticleSoA<FloatType>> _particles{std::nullopt};
        const cl_uint _numParticles;
        const cl_scalar _deltaT;
        const cl_vec4 _globalForce;

        cl_device_id gpu{nullptr};
        cl_context context{nullptr};
        cl_command_queue queue{nullptr};
        cl_program program{nullptr};
        cl_kernel kernelPositionUpdate{nullptr};
        cl_kernel kerneVelocityUpdate{nullptr};
        cl_kernel kernelForceUpdate{nullptr};

    public:

        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplOpenCL(const ParticleSimulationConfig<FloatType> &config);

        ~ImplOpenCL();

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