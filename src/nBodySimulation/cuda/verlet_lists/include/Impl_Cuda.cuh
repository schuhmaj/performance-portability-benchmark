#pragma once

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include "CudaParticleSoA.cuh"

namespace ppb {
    template <typename FloatType>
    class ImplCuda {
        int _blockSize;
        int _gridSize;
        size_t frequency;
        size_t size;
        size_t iteration{0};

        /**
         * @brief 'starts' looks like this:
         * 0, 2, 2, 4, ..., 10
         * The first list has 2 neighbors, the second has none, the third has 2 neighbors again.
         * The stored indicies are the indicies in the 'verletLists' container, where the i-th index 
         * describes the starting index of the neighbor list of the i-th particle inside 'verletLists'.
         */
        int* starts{nullptr};
        /**
         * @brief 'cells' contains the sorted particle *indicies* to the particles contained in 'particles'.
         * 'starts' marks the start of each cell.
         */
        int* verletLists{nullptr};
        
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
        
        ~ImplCuda();
    };

} // namespace ppb
