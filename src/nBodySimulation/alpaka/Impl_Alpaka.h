#pragma once
#include <vector>
#include <array>
#include <utility>
#include <optional>
#include "alpaka/alpaka.hpp"
#include "alpaka/example/ExecuteForEachAccTag.hpp"
#include "alpaka/example/ExampleDefaultAcc.hpp"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"


namespace ppb {

    /** Dimensionality of the Problem */
    using Dim = alpaka::DimInt<1u>;
    /** The Integer Type used for indexing and sizes **/
    using Idx = std::size_t;
    /** The Host Backend, Serial CPU **/
    using Host = alpaka::DevCpu;
    /** Defines the Compute Backend/ Device to use; We chose the first one which is enabled; CUDA/ HIP devices have precedence in this "ExampleDefault" List **/
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    /** Defines the Runtime of the chosen Accelerator, i.e. CUDA, the software layer **/
    using Platform = alpaka::Platform<Acc>;
    /** Defines the actual physical device, i.e. RTX 2080 **/
    using Device = alpaka::Dev<Platform>;
    /** The Compute Pipline for the Accelerator Device **/
    using Queue = alpaka::Queue<Device, alpaka::Blocking>;

    template <typename FloatType>
    class AlpakaParticleSoA {
        const std::vector<Particle<FloatType>> &_ref;

    public:
        using float_type = FloatType;
        using BufHost2D = alpaka::Buf<Host, float_type, Dim, Idx>;
        using BufAcc2D = alpaka::Buf<Device, float_type, Dim, Idx>;

        alpaka::Vec<Dim, Idx> extent;
        BufAcc2D positions;
        BufAcc2D velocities;
        BufAcc2D forces;
        BufAcc2D oldForces;

        BufHost2D positionsHost;
        BufHost2D velocitiesHost;
        BufHost2D forcesHost;

        AlpakaParticleSoA(const std::vector<Particle<FloatType>> &ref, Host &host, Device &device, Queue &queue);
        std::vector<Particle<FloatType>> toParticles(Queue &queue);
    };

    template<typename FloatType>
    class ImplAlpaka {
        ParticleSimulationConfig<FloatType> _config;
        ParticleSimulationTimings _timings;
        std::optional<AlpakaParticleSoA<FloatType>> _particles{std::nullopt};

    public:
        using float_type = FloatType;

        /** The CPU/ Host **/
        Host host;
        /** The actual device chosen by ID, using the runtime, i.e. CUDA_DEVICE 0, 1, ... **/
        Device device;
        /** The compute pipeline for the chosen device **/
        Queue queue;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplAlpaka(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Runs the simulation for the configured total simulation time, performing position, force,
         * and velocity updates for each step using the velocity Verlet scheme.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::vector<Particle<FloatType>> Final state of all particles after the simulation.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

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

    };
