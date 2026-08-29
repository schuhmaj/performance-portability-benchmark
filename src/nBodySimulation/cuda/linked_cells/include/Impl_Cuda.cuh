#pragma once

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include "CudaParticleSoA.cuh"

namespace ppb::cuda::nbody {
    template <typename FloatType>
    class ImplCuda {
        /**
        * The block size used for all kernels, except the force computation kernel
        */
        int _blockSize;
       
        /**
        * The grid size used for all kernels, except the force computation kernel
        */
        int _gridSize;
        
        /**
        * The block size used for the force computation kernels
        */
        int _blockSizeForces;
     
        /**
        * The grid size used for the force computation kernels
        */
        int _gridSizeForces;
      
        /**
        * A copy of X_DIM (number of cells in x-dimension of domain) on host memory
        */
        int x_dim_h;
       
        /**
        * A copy of Y_DIM (number of cells in y-dimension of domain) on host memory
        */
        int y_dim_h;
       
        /**
        * A copy of Z_DIM (number of cells in z-dimension of domain) on host memory
        */
        int z_dim_h;

        /**
         * An array indicating where each cell starts and ends in 'cells'.
         * 'starts' looks like this:
         * 0, 2, 2, 4, ..., 10
         * The first cell has 2 particles, the second has none, the third has 2 particles again.
         * The stored indicies are the indicies in the 'cells' container, where the i-th index 
         * describes the starting index of the i-th cell inside 'cells'.
         */
        int* starts{nullptr};
        
        /**
         * 'cells' contains the sorted particle *indicies* to the particles contained in 'positions', 'velocities', 'forces' in CudaParticleSoA.
         * 'starts' marks the start of each cell.
         */
        int* cells{nullptr};

        /**
        * Array used when updating the cells. This array stores the offset of a given particle within its designated cell to avoid race conditions.
        */
        int* cell_offsets{nullptr};
       
        /**
        * Array that stores the sorted particles' positions.
        */
        float3* cells_positions{nullptr};

        /**
        * Permanent array that is used in update_cells.
        */
        int* tmp{nullptr};

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

} // namespace ppb::cuda::nbody
