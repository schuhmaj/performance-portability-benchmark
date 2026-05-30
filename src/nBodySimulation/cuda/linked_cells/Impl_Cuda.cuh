#pragma once

#include <thrust/device_vector.h>
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {

    template <typename FloatType>
    struct CudaParticleSoA {
        __constant__ float x_dim;
        __constant__ float y_dim;
        __constant__ float z_dim;
        __constant__ std::vector<Particle<FloatType>> &_ref;
        __constant__ float cell_size{1.0f};
        __constant__ float cutoff_radius{1.0f};
        __constant__ std::array<size_t, 27> offsets;

        /**
         * @brief 'starts' looks like this:
         * 0, 2, 2, 4, ..., 10
         * so here for example we have 10 cells (as indicated by the last entry). 
         * The first one has 2 particles, the second has none, the third has 2 particles again.
         * The stored indicies are the indicies in the 'cells' container, where the i-th index describes the starting
         * 
         */
        __device__ size_t* starts{nullptr};
        __device__ size_t* cells{nullptr};
        /**
         * @brief 'cells' contains the sorted particle *indicies* to the particles contained in 'particles'.
         * 'starts' marks the start of each cell.
         */
        __device__ float3* positions{nullptr};
        __device__ float3* velocities{nullptr};
        __device__ float3* forces{nullptr};
        __device__ float3* oldForce{nullptr};
        
        __host__ std::vector<float3> positionsHost;
        __host__ std::vector<float3> velocitiesHost;
        __host__ std::vector<float3> forcesHost;

        explicit CudaParticleSoA(const std::vector<Particle<FloatType>> &particles);

        ~CudaParticleSoA();

        std::vector<Particle<FloatType>> toParticles();
    };

    template <typename FloatType>
    class ImplCuda {
        __constant__ int _blockSize;
        __constant__ int _gridSize;
        __constant__ float3 _globalForce;

        ParticleSimulationConfig<FloatType> _config;

        std::optional<CudaParticleSoA<FloatType>> _particles{std::nullopt};

        ParticleSimulationTimings _timings{};
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
