#pragma once

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include "CudaParticleSoA.cuh"

namespace ppb::cuda::nbody {
    struct BoundingBox {
        float3 lowerCorner; //corner of bounding box with min x, y, z coordinates
        float3 upperCorner; //corner of bounding box with max x, y, z coordinates
    };
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
        
        // FOR NOW THE ONLY SUPPORTED M AND N ARE M = 8 AND N = 4!!! OTHER VALUES WILL LEAD TO UNDEFINED BEHAVIOUR!!!
        int M = 8; //M must be a multiple of N
        int N = 4;
        int* starts_towers{nullptr};        //contains the indicies in 'clusters' where the towers start
        int* clusters{nullptr};             //towered + binned particles. Contains references to the particles. If the reference is -1, then that particle is a dummy particle.
        float* z_coordinates{nullptr};      //the z-coordinates of the particles in the cluster. Needed for sorting along the z-dimension.
        BoundingBox* BBM{nullptr};          //k-th entry is bounding box of k-th i-cluster (which has size M)
        BoundingBox* BBN{nullptr};          //k-th entry is bounding box of k-th j-cluster (which has size N)
        int* cluster_pairs{nullptr};        //boundaries denoted by 'starts'
        size_t num_towers = 0;
        float tower_size = 0.f;
        size_t size_clusters = 0;

        /**
         * @brief 'verletLists' is a concatenation of all verlet lists. 
         * It contains the particle *indicies* to the particles contained in 'particles'.
         * 'starts' marks the start of each list.
         */
        int* verletLists{nullptr};
         
        int* starts_LC{nullptr};
        int* cells{nullptr};
        int* cell_offsets{nullptr};
        int* tmp{nullptr};
        size_t num_cells = 0;
        
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

        void makeClusters();

        void boundingBoxes();

        void createPairList();
        
        ~ImplCuda();
    };

} // namespace ppb::cuda::nbody
