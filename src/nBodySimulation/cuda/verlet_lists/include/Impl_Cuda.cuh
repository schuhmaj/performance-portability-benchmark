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
        /**
        * The block size used for all kernels, except the force computation kernel of Verlet Cluster Lists
        */
        int _blockSize;
        
        /**
        * The grid size used for all kernels, except the force computation kernel
        */
        int _gridSize;
        
        /**
        * The block size used for the force computation kernel of Verlet Cluster Lists 
        */
        int _blockSizeForces;

        /**
        * The neighbor update interval (or frequency)
        */
        size_t frequency;

        /**
        * The number of particles
        */
        size_t size;

        /**
        * The current simulation iteration
        */
        size_t iteration{0};

        /**
         * FOR VERLET LISTS
         * An array indicating where each neighbor list starts and ends in 'verletLists'.
         * 'starts' looks like this:
         * 0, 2, 2, 4, ..., 10
         * The first particle has 2 particles in the neighbor list, the second has none, the third has 2 particles again.
         * The stored indicies are the indicies in the 'verletLists' container, where the i-th index 
         * describes the starting index of the i-th neighbor list inside 'verletLists'.
         *
         * FOR VERLET CLUSTER LISTS
         * 'starts' works exactly the same but instead of storing the start and end indices of 'verletLists' it stores 
         * those of 'cluster_pairs'.
         */
        int* starts{nullptr};
        
//------------------------------------------ VERLET CLUSTER LISTS ----------------------------------------------
        // FOR NOW THE ONLY SUPPORTED M AND N ARE M = 8 AND N = 4!!! OTHER VALUES WILL LEAD TO UNDEFINED BEHAVIOUR!!!
        /**
        * Size of the i-clusters (FOR NOW ONLY M = 8 IS SUPPORTED!!!)
        */
        int M = 8; //M must be a multiple of N

        /**
        * Size of the j-clusters (FOR NOW ONLY N = 4 IS SUPPORTED AND M MUST BE A MULTIPLE OF N!!!)
        */
        int N = 4;

        /**
        * Works similarly to 'starts' but instead contains the indices in 'clusters' where the towers start
        */
        int* starts_towers{nullptr};

        /**
        * The towered + binned particles. Contains references to the particles. If the reference is -1, then that particle is a dummy particle
        */
        int* clusters{nullptr};

        /**
        * The z-coordinates of the particles in the cluster. Needed for sorting along the z-dimension
        */
        float* z_coordinates{nullptr};

        /**
        * The k-th element is the bounding box of the k-th i-cluster (which has size M)
        */
        BoundingBox* BBM{nullptr};

        /**
        * The k-th element is the bounding box of the k-th j-cluster (which has size N)
        */
        BoundingBox* BBN{nullptr};

        /**
        * The pair list of clusters. Contains only the indices of the clusters. Boundaries are denoted by 'starts'
        */
        int* cluster_pairs{nullptr};

        /**
        * The total number of towers
        */
        size_t num_towers = 0;

        /**
        * The side-length of each tower. The sidelength in x and y is always the same.
        */
        float tower_size = 0.f;

        /**
        * The total number of i-clusters, counting those clusters that have dummy particles in them.
        */
        size_t size_clusters = 0;

//------------------------------------------------- VERLET LISTS --------------------------------------------
        /**
         * 'verletLists' is a concatenation of all verlet lists. 
         * It contains the particle *indicies* to the particles contained in 'particles'.
         * 'starts' marks the start of each list.
         */
        int* verletLists{nullptr};
        
//---------------------------------------------- VERLET LISTS LC OPT--------------------------------------------
        /**
        * Works like 'starts' but indicates the start and end indices of each cell in 'cells'
        */
        int* starts_LC{nullptr};

        /**
         * 'cells' contains the sorted particle *indicies* to the particles contained in 'positions', 'velocities', 'forces' in CudaParticleSoA.
         * 'starts_LC' marks the start of each cell.
         */
        int* cells{nullptr};

        /**
        * Array used when updating the cells. This array stores the offset of a given particle within its designated cell to avoid race conditions.
        */
        int* cell_offsets{nullptr};

        /**
        * Permanent array that is used in update_cells.
        */
        int* tmp{nullptr};

        /**
        * The total number of cells in the simulation domain
        */
        size_t num_cells = 0;
//--------------------------------------------------------------------------------------------------------------------- 
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

        /**
        * Groups the particles into clusters by first binning them into towers based on their x- and y-coordinates 
        * and then sorting them along the z-axis.
        */
        void makeClusters();

        /**
        * Computes the bounding boxes of the i- and j-clusters.
        */
        void boundingBoxes();

        /**
        * Populates the cluster pair list 'cluster_pairs' by running the neighbor cluster search.
        */
        void createPairList();
        
        ~ImplCuda();
    };

} // namespace ppb::cuda::nbody
