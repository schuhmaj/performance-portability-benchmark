#pragma once

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {


    template <typename FloatType>
    struct CudaParticleSoA {

        const std::vector<Particle<FloatType>> &_ref;

        float3* positions{nullptr};
        float3* velocities{nullptr};
        float3* forces{nullptr};
        float3* oldForces{nullptr};
        /**
         * Other solutions possible but either:
         * 1) more memory -> not ideal for GPU cause it's compute optimized
         * 2) no more memory coalescence -> less cache hits, probably not worth the slight computational upgrade of not having the starts vector anymore
         */

        /**
         * @brief 'starts' looks like this:
         * 0, 2, 2, 4, ..., 10
         * so here for example we have 10 cells (as indicated by the last entry). 
         * The first one has 2 particles, the second has none, the third has 2 particles again.
         * The stored indicies are the indicies in the 'cells' container, where the i-th index describes the starting
         * 
         */
        std::vector<size_t>* starts{nullptr};
        /**
         * IMPORTANT: 'cells' can only contain up to SIZE_T_MAX - 1 cells. SIZE_T_MAX is a reserved special value.
         */
        std::vector<size_t>* cells{nullptr};
        
        std::vector<float3> positionsHost;
        std::vector<float3> velocitiesHost;
        std::vector<float3> forcesHost;
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
