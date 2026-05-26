#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {

    template <typename FloatType>
    struct CudaParticleSoA {
        const float x_dim;
        const float y_dim;
        const float z_dim;
        const std::vector<Particle<FloatType>> &_ref;

        //size_t for index of particle in original 'particles' container, float3 for {x, y, z} position.
        thrust::device_vector<thrust::device_vector<std::pair<size_t, float3>>> positions;
        thrust::device_vector<float3> velocities;
        thrust::device_vector<float3> forces;
        thrust::device_vector<float3> oldForces;
         
        thrust::host_vector<float3> positionsHost;
        thrust::host_vector<float3> velocitiesHost;
        thrust::host_vector<float3> forcesHost;
        float cell_size{1.0f}; //no support for non-square cells (for now)
        float cutoff_radius{1.0f};

        explicit CudaParticleSoA(const std::vector<Particle<FloatType>> &particles);

        ~CudaParticleSoA();

        std::vector<Particle<FloatType>> toParticles();
    };

    template <typename FloatType>
    class ImplCuda {

        ParticleSimulationConfig<FloatType> _config;

        std::optional<CudaParticleSoA<FloatType>> _particles{std::nullopt};

        ParticleSimulationTimings _timings{};

        int _blockSize;
        int _gridSize;
        float3 _globalForce;

    public:
        using float_type = FloatType;


        explicit ImplCuda(const ParticleSimulationConfig<FloatType> &config);

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
