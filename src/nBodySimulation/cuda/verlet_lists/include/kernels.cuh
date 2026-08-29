#pragma once 

#include <float.h>
#include <stdio.h>
#include "Impl_Cuda.cuh"
#include "constants.cuh"
#include "common/cuda/Cuda_Float3_Arithmetic.cuh"

namespace ppb::cuda::nbody {
    //taken from: https://github.com/dangets/cuda_examples/blob/master/clamp_function.cu (last accessed 12.6.26) 
    /**
    * @brief std::clamp implementation that can be used in device code. 
    * @param val The value to be clamped
    * @param vMin The min value that val is clamped to if it is smaller
    * @param vMax The max value that val is clamped to if it is larger
    */
    template <typename T>
    __device__ inline T clamp(T val, T vMin, T vMax) {
        return min(max(val, vMin), vMax);
    }

    /**
    * @brief Updates the positions of the particles
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    * @param velocities The velocities buffer. The i-th element is the velocity of the i-th particle
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param oldForces The buffer storing the previous iteration's forces. The i-th element is the force of the previous iterations of the i-th particle
    */
    __global__ void update_positions(
        float3* positions, 
        const float3* velocities, 
        float3* forces, 
        float3* oldForces 
    );

    /**
    * @brief Updates the velocities of the particles
    * @param velocities The velocities buffer. The i-th element is the velocity of the i-th particle
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param oldForces The buffer storing the previous iteration's forces. The i-th element is the force of the previous iterations of the i-th particle
    */
    __global__ void update_velocities(
        float3* velocities, 
        const float3* forces, 
        const float3* oldForces
    );

    /**
    * @brief [VERLET LISTS] Updates the forces of the particles.
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param verletLists The flattened list of all neighbor lists
    * @param starts The starts buffer that indicates where in 'verletLists' the different neighbor lists start and end
    */
    __global__ void compute_forces(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ verletLists,
        const int* __restrict__ starts
    );

    /**
    * @brief [VERLET LISTS NAIVE] Determines the total number of neighbors in the current simulation step
    * @param starts The buffer that the histogram of neighbors will be stored in
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    */
    __global__ void get_number_of_neighbors(
        int* starts, 
        float3* positions
    );

    /**
    * @brief [VERLET LISTS NAIVE] Populates the 'verletLists' buffer
    * @param verletLists The flattened list of all neighbor lists that will be populated here
    * @param starts The starts buffer that indicates where in 'verletLists' the different neighbor lists start and end
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    */
    __global__ void make_verlet_lists(
        int* verletLists, 
        int* starts, 
        float3* positions
    );

//--------------------------------------------- VERLET CLUSTER LISTS --------------------------------------------------
    /**
    * @brief [VERLET CLUSTER LISTS] Returns the tower id of the tower that a given particle is contained inside of
    * @param particle_idx The index of the particle in 'positions'
    * @param positions The positions of all the particles
    * @param grid_size The size of the tower-grid (i.e. the side length of the towers)
    * @returns The tower id of the tower that a given particle is contained inside of
    */ 
    __device__ inline int get_tower_id(int particle_idx, float3* __restrict__ positions, float grid_size) {
        int x_dim = util::ceilDiv((BOX_MAX[0] - BOX_MIN[0]), grid_size);
        int y_dim = util::ceilDiv((BOX_MAX[1] - BOX_MIN[1]), grid_size);
        int tower_x = clamp<int>(int(((positions[particle_idx].x - BOX_MIN[0]) / grid_size)), 0, x_dim - 1);
        int tower_y = clamp<int>(int(((positions[particle_idx].y - BOX_MIN[1]) / grid_size)), 0, y_dim - 1);
        return tower_x + (tower_y * x_dim);
    }

    /**
    * @brief [VERLET CLUSTER LISTS] Computes the tower id for each particle
    * @param positions The positions of all the particles
    * @param particle_tower_id The buffer where the tower ids of the various particles will be stored
    * @param grid_size The size of the tower-grid (i.e. the side length of the towers)
    */
    __global__ void get_tower_id_per_particle(
        float3* __restrict__ positions,
        size_t* __restrict__ particle_tower_id,
        float grid_size
    );

    /**
    * @brief [VERLET CLUSTER LISTS] Determines the number of particles per tower
    * @param positions The positions of all the particles
    * @param starts_towers The buffer where the histogram of particles per towers will be stored
    * @param grid_size The size of the tower-grid (i.e. the side length of the towers)
    */
    __global__ void count_particles_in_towers(
        float3* __restrict__ positions, 
        int* __restrict__ starts_towers,
        float grid_size
    );

    /**
    * @brief [VERLET CLUSTER LISTS] Computes the number of dummy particles required per tower
    * @param starts_towers The histogram of particles per towers
    * @param cluster_size The size of the i-clusters (here always 8)
    * @param num_towers The total number of towers
    */
    __global__ void add_dummy_particles_to_towers(
        int* __restrict__ starts_towers, 
        int cluster_size, 
        int num_towers
    );

    /**
    * @brief [VERLET CLUSTER LISTS] Determines the exact position of the particle in the 'clusters' array. Neccessary to avoid race conditions.
    * @param positions The positions of all the particles
    * @param clusters The buffer where the clustered particles will be stored1
    * @param starts_towers The histogram of particles per towers
    * @param positions_in_towers Buffer where the offsets per tower is stored
    * @param grid_size The size of the tower-grid (i.e. the side length of the towers)
    */
    __global__ void get_particle_position_in_tower(
        float3* __restrict__ positions, 
        int* __restrict__ clusters, 
        int* __restrict__ starts_towers,
        int* __restrict__ positions_in_tower,
        float grid_size
    );

    /**
    * @brief [VERLET CLUSTER LISTS] Inserts the clustered particles into 'clusters'
    * @param positions The positions of all the particles
    * @param clusters The buffer where the clustered particles will be stored
    * @param z_coordinates The z-coordinates of the particles
    * @param starts_towers The histogram of particles per towers
    * @param positions_in_towers Buffer where the offsets per tower is stored
    * @param grid_size The size of the tower-grid (i.e. the side length of the towers)
    */
    __global__ void insert_particles_into_towers(
        float3* __restrict__ positions, 
        int* __restrict__ clusters, 
        float* __restrict__ z_coordinates,
        int* __restrict__ starts_towers,
        int* __restrict__ positions_in_tower,
        float grid_size
    );

    /**
    * @brief [VERLET CLUSTER LISTS] Initializes the 'clusters' and 'z_coordinates' buffers
    * @param clusters The buffer where the clustered particles will be stored
    * @param z_coordinates The z-coordinates of the particles
    * @param size_clusters The total number of of clusters
    */
    __global__ void init_clusters_and_z_coordinates(
        int* __restrict__ clusters, 
        float* __restrict__ z_coordinates,
        int size_clusters
    );

    /**
    * @brief [VERLET CLUSTER LISTS] Computes the bounding boxes for a given set of clusters
    * @param BB The buffer where the bounding boxes will be stored
    * @param clusters The buffer where the clustered particles will be stored
    * @param positions The positions of all the particles
    * @param cluster_size The size of the i-clusters (here always 8)
    * @param num_towers The total number of towers
    */
    __global__ void compute_bounding_boxes(
        BoundingBox* __restrict__ BB, 
        int* __restrict__ clusters, 
        float3* __restrict__ positions,
        int cluster_size,
        int num_clusters
    );

    /**
    * @brief [VERLET CLUSTER LISTS] Computes the component-wise maximum between two float3 vectors
    * Inspired by AutoPas: https://github.com/AutoPas/AutoPas/blob/master/src/autopas/utils/ArrayMath.h
    * @param a
    * @param b
    * @returns The component-wise maximum between a and b
    */
    __device__ inline float3 maxF3(const float3& a, const float3& b) {
        return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
    }

    /**
    * @brief [VERLET CLUSTER LISTS] Computes the squared distance between two bounding boxes
    * Inspired by AutoPas: https://github.com/AutoPas/AutoPas/blob/af9a1530fca6543aa651600751256cf408deaf13/src/autopas/containers/verletClusterLists/VerletClusterListsRebuilder.h
    * @param a
    * @param b 
    * @returns The squared distance between bounding box a and bounding box b
    */
    __device__ inline float BBdistanceSquared(BoundingBox& a, BoundingBox& b) {
        float3 aToB = maxF3(make_float3(0.f, 0.f, 0.f), make_float3_sub(a.lowerCorner, b.upperCorner));
        float3 bToA = maxF3(make_float3(0.f, 0.f, 0.f), make_float3_sub(b.lowerCorner, a.upperCorner));
        return dot3(aToB, aToB) + dot3(bToA, bToA);
    }

    /**
    * Order of N3L: iterate over all x from 0 to x_dim - 1 in one y-dimension, then repeat on the next *higher* y-dimension
    * It's basically like a zig-zag motion from the lower-left corner of the domain to the upper-right corner.
    * @brief [VERLET CLUSTER LISTS] Determines if a given neighbor tower is a forward neighbor to a given tower.
    * @param neighbor_x The x-coordinate of the neighbor tower
    * @param neighbor_y The y-coordinate of the neighbor tower
    * @param tower_idx_x The x-coordinate of the base tower
    * @param tower_idx_y The y-coordinate of the base tower
    * @returns true if neighbor tower is a forward tower, false otherwise
    */
    __device__ inline bool isForwardNeighbor(int neighbor_x, int neighbor_y, int tower_idx_x, int tower_idx_y) {
        return (neighbor_x > tower_idx_x && neighbor_y >= tower_idx_y) || neighbor_y > tower_idx_y || (neighbor_x == tower_idx_x && neighbor_y == tower_idx_y);
    }

    /**
    * @brief [VERLET CLUSTER LISTS UNOPTIMIZED] Performs the unoptimized cluster pair search
    * @param BBM The array of all i-clusters
    * @param BBN The array of all j-clusters
    * @param count Boolean flag to indicate whether the kernel shall be used to only count the number of neighbors, or actually populate 'cluster_pairs'
    * @param cluster_pairs The pair list of clusters
    * @param starts The array that will mark the starts and ends of the various cluster lists
    * @param clusters The buffer of the clustered particles
    * @param positions The positions of all the particles
    * @param grid_size The size of the tower-grid (i.e. the side length of the towers)
    * @param size_clusters The total number of of clusters
    */
    __global__ void cluster_pair_search(
        BoundingBox* __restrict__ BBM,
        BoundingBox* __restrict__ BBN,
        bool count,
        int* __restrict__ cluster_pairs, 
        int* __restrict__ starts,
        int* __restrict__ clusters,
        float3* __restrict__ positions,
        float grid_size,
        int size_clusters
    );
   
    /**
    * @brief [VERLET CLUSTER LISTS OPTIMIZED] Performs the optimized cluster pair search
    * @param BBM The array of all i-clusters
    * @param BBN The array of all j-clusters
    * @param count Boolean flag to indicate whether the kernel shall be used to only count the number of neighbors, or actually populate 'cluster_pairs'
    * @param cluster_pairs The pair list of clusters
    * @param starts The array that will mark the starts and ends of the various cluster lists
    * @param clusters The buffer of the clustered particles
    * @param positions The positions of all the particles
    * @param grid_size The size of the tower-grid (i.e. the side length of the towers)
    * @param size_clusters The total number of of clusters
    */
    __global__ void cluster_pair_search_optimized(
        BoundingBox* __restrict__ BBM,
        BoundingBox* __restrict__ BBN,
        bool count,
        int* __restrict__ cluster_pairs, 
        int* __restrict__ starts,
        int* __restrict__ starts_towers,
        int* __restrict__ clusters,
        float3* __restrict__ positions,
        float grid_size,
        int size_clusters
    );
    
    /**
    * @brief [VERLET CLUSTER LISTS] Performs the force computation of Verlet Cluster Lists
    * @param positions The positions of all the particles
    * @param forces The forces of all the particles
    * @param clusters The buffer of the clustered particles
    * @param cluster_pairs The pair list of clusters
    * @param starts The array that will mark the starts and ends of the various cluster lists
    * @param size_clusters The total number of of clusters
    */
    __global__ void compute_force_cluster_lists(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ clusters,
        const int* __restrict__ cluster_pairs,
        const int* __restrict__ starts, //starts for cluster_pairs
        int size_clusters
    );

//--------------------------------------------- VERLET LISTS LC OPTIMIZATION --------------------------------------------------
    /**
    * @brief Returns the index of the cell that a given particle is contained inside of
    * @param particle_idx The index of the particle in 'positions'
    * @param positions The array storing the positions of the particles
    * @returns The index of the cell that the particle at particle_idx is contained inside of
    */
    __device__ inline int get_cell_idx(size_t particle_idx, const float3* positions) {
        int x_idx = clamp<int>(int(((positions[particle_idx].x - BOX_MIN[0]) / CELL_SIZE)), 0, X_DIM - 1);
        int y_idx = clamp<int>(int(((positions[particle_idx].y - BOX_MIN[1]) / CELL_SIZE)), 0, Y_DIM - 1);
        int z_idx = clamp<int>(int(((positions[particle_idx].z - BOX_MIN[2]) / CELL_SIZE)), 0, Z_DIM - 1);
        return x_idx + (y_idx * X_DIM) + (z_idx * X_DIM * Y_DIM); 
    }

    /**
    * @brief Checks if a cell at a given offset from a given index of another cell is within the bounds of the simulation domain.
    * @param idx The index from which the offset starts
    * @param offset The offset in indices that the other cell has from the cell at idx
    * @returns true if offset within bounds, false otherwise
    */
    __device__ inline bool is_in_bounds(int idx, int offset) {
        int x_idx = idx % X_DIM;
        int y_idx = (idx / X_DIM) % Y_DIM;
        int z_idx = (idx / (X_DIM * Y_DIM));

        int offset_x = OFFSETS_XYZ[3 * offset];
        int offset_y = OFFSETS_XYZ[3 * offset + 1];
        int offset_z = OFFSETS_XYZ[3 * offset + 2];

        if (x_idx + offset_x < 0 || x_idx + offset_x > X_DIM - 1) return false;
        if (y_idx + offset_y < 0 || y_idx + offset_y > Y_DIM - 1) return false;
        if (z_idx + offset_z < 0 || z_idx + offset_z > Z_DIM - 1) return false;
        return true;
    }

    /**
    * @brief Sorts the particles into cells
    * @param positions The positions of all the particles    
    * @param tmp A temporary buffer where the i-th element is the index of the cell that the i-th particle is contained inside of
    * @param cell_offsets Buffer storing the offsets of the particles within their respective cell.
    * @param starts_LC The starts buffer that indicates where in 'cells' the different cells start and end
    */
    __global__ void sort_particles_into_cells(
        float3* positions, 
        int* tmp, 
        int* cell_offsets,
        int* starts_LC
    );

    /**
    * @brief Sorts the particles into cells
    * @param cells The sorted particles in the cells
    * @param tmp A temporary buffer where the i-th element is the index of the cell that the i-th particle is contained inside of
    * @param cell_offsets Buffer storing the offsets of the particles within their respective cell.
    * @param starts_LC The starts buffer that indicates where in 'cells' the different cells start and end
    * @param positions The positions of all the particles    
    */
    __global__ void update_cells(
        int* cells, 
        int* tmp, 
        int* cell_offsets, 
        int* starts_LC,
        float3* positions
    );

    /**
    * @brief Sorts the particles into cells
    * @param starts The starts buffer that indicates where in 'verletLists' the different neighbor lists start and end
    * @param positions The positions of all the particles    
    * @param starts_LC The starts buffer that indicates where in 'cells' the different cells start and end
    * @param cells The sorted particles in the cells
    */
    __global__ void get_number_of_neighbors_LC_OPT(
        int* starts, 
        float3* positions, 
        int* starts_LC, 
        int* cells
    );
    
    /**
    * @brief Sorts the particles into cells
    * @param verletLists The flattened list of all neighbor lists
    * @param starts The starts buffer that indicates where in 'verletLists' the different neighbor lists start and end
    * @param positions The positions of all the particles    
    * @param starts_LC The starts buffer that indicates where in 'cells' the different cells start and end
    * @param cells The sorted particles in the cells
    */
    __global__ void make_verlet_lists_LC_OPT(
        int* verletLists, 
        int* starts, 
        float3* positions, 
        int* starts_LC, 
        int* cells
    );
} // namespace ppb::cuda::nbody