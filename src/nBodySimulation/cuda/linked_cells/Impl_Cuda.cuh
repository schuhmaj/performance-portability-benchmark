#pragma once

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {

    template <typename FloatType>
    struct CudaParticleSoA {
        const std::vector<Particle<FloatType>> &_ref;

        //Device memory
        float3* positions{nullptr};
        float3* velocities{nullptr};
        float3* forces{nullptr};
        float3* oldForces{nullptr};
        
        //Host memory
        std::vector<float3> positionsHost;
        std::vector<float3> velocitiesHost;
        std::vector<float3> forcesHost;

        explicit CudaParticleSoA(const std::vector<Particle<FloatType>> &particles);

        ~CudaParticleSoA();

        std::vector<Particle<FloatType>> toParticles();
    };

    template <typename FloatType>
    class ImplCuda {
        int _blockSize;
        int _gridSize;
        float3 _globalForce;
        int x_dim;
        int y_dim;
        int z_dim;
        int offsets[27];
#ifdef PPB_ENABLE_DOMAIN_COLORING
        int _gridSizeColored;
        int _blockSizeColored;
        int offsets_colored[8];
#endif

        /**
         * @brief 'starts' looks like this:
         * 0, 2, 2, 4, ..., 10
         * The first one has 2 particles, the second has none, the third has 2 particles again.
         * The stored indicies are the indicies in the 'cells' container, where the i-th index 
         * describes the starting index of the i-th cell inside 'cells'.
         */
        int* starts{nullptr};
        /**
         * @brief 'cells' contains the sorted particle *indicies* to the particles contained in 'particles'.
         * 'starts' marks the start of each cell.
         */
        int* cells{nullptr};
       
        int* cell_offsets{nullptr};

        /**
        * @brief Permanent array that is used in update_cells. (TODO: replace this with cooperative groups!!)
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

} // namespace ppb
