#pragma once 

#include <float.h>
#include <stdio.h>
#include "Impl_Cuda.cuh"
#include "constants.cuh"
#include "common/cuda/Cuda_Float3_Arithmetic.cuh"

namespace ppb::cuda::nbody {
    //taken from: https://github.com/dangets/cuda_examples/blob/master/clamp_function.cu (last accessed 12.6.26)
    template <typename T>
    __device__ inline T clamp(T val, T vMin, T vMax) {
        return min(max(val, vMin), vMax);
    }

    __global__ void update_positions(
        float3* positions, 
        const float3* velocities, 
        float3* forces, 
        float3* oldForces 
    );

    __global__ void update_velocities(
        float3* velocities, 
        const float3* forces, 
        const float3* oldForces
    );

    __global__ void compute_forces(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ verletLists,
        const int* __restrict__ starts
    );

    __global__ void get_number_of_neighbors(
        int* starts, 
        float3* positions
    );

    __global__ void make_verlet_lists(
        int* verletLists, 
        int* starts, 
        float3* positions
    );

//--------------------------------------------- VERLET CLUSTER LISTS --------------------------------------------------
    __device__ inline int get_tower_id(int particle_idx, float3* __restrict__ positions, float grid_size) {
        int x_dim = util::ceilDiv((BOX_MAX[0] - BOX_MIN[0]), grid_size);
        int y_dim = util::ceilDiv((BOX_MAX[1] - BOX_MIN[1]), grid_size);
        int tower_x = clamp<int>(int(((positions[particle_idx].x - BOX_MIN[0]) / grid_size)), 0, x_dim - 1);
        int tower_y = clamp<int>(int(((positions[particle_idx].y - BOX_MIN[1]) / grid_size)), 0, y_dim - 1);
        return tower_x + (tower_y * x_dim);
    }

    __global__ void get_tower_id_per_particle(
        float3* __restrict__ positions,
        size_t* __restrict__ particle_tower_id,
        float grid_size
    );

    __global__ void count_particles_in_towers(
        float3* __restrict__ positions, 
        int* __restrict__ starts_towers,
        float grid_size
    );

    __global__ void add_dummy_particles_to_towers(
        int* __restrict__ starts_towers, 
        int cluster_size, 
        int num_towers
    );

    __global__ void get_particle_position_in_tower(
        float3* __restrict__ positions, 
        int* __restrict__ clusters, 
        int* __restrict__ starts_towers,
        int* __restrict__ positions_in_tower,
        float grid_size
    );

    __global__ void insert_particles_into_towers(
        float3* __restrict__ positions, 
        int* __restrict__ clusters, 
        float* __restrict__ z_coordinates,
        int* __restrict__ starts_towers,
        int* __restrict__ positions_in_tower,
        float grid_size
    );

    __global__ void init_clusters_and_z_coordinates(
        int* __restrict__ clusters, 
        float* __restrict__ z_coordinates,
        int size_clusters
    );

    __global__ void compute_bounding_boxes(
        BoundingBox* __restrict__ BB, 
        int* __restrict__ clusters, 
        float3* __restrict__ positions,
        int cluster_size,
        int num_clusters
    );

    //Inspired by AutoPas: 
    //https://github.com/AutoPas/AutoPas/blob/master/src/autopas/utils/ArrayMath.h
    //https://github.com/AutoPas/AutoPas/blob/af9a1530fca6543aa651600751256cf408deaf13/src/autopas/containers/verletClusterLists/VerletClusterListsRebuilder.h
    __device__ inline float3 maxF3(const float3& a, const float3& b) {
        return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
    }

    __device__ inline float BBdistanceSquared(BoundingBox& a, BoundingBox& b) {
        float3 aToB = maxF3(make_float3(0.f, 0.f, 0.f), make_float3_sub(a.lowerCorner, b.upperCorner));
        float3 bToA = maxF3(make_float3(0.f, 0.f, 0.f), make_float3_sub(b.lowerCorner, a.upperCorner));
        return dot3(aToB, aToB) + dot3(bToA, bToA);
    }

    /** Order of N3L: iterate over all x from 0 to x_dim - 1 in one y-dimension, then repeat on the next *higher* y-dimension
    * It's basically like a zig-zag motion from the lower-left corner of the domain to the upper-right corner. */
    __device__ inline bool isForwardNeighbor(int neighbor_x, int neighbor_y, int tower_idx_x, int tower_idx_y) {
        return (neighbor_x > tower_idx_x && neighbor_y >= tower_idx_y) || neighbor_y > tower_idx_y || (neighbor_x == tower_idx_x && neighbor_y == tower_idx_y);
    }

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
    
    __global__ void compute_force_cluster_lists(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ clusters,
        const int* __restrict__ cluster_pairs,
        const int* __restrict__ starts, //starts for cluster_pairs
        int size_clusters
    );

//--------------------------------------------- VERLET LISTS LC OPTIMIZATION --------------------------------------------------
    __device__ inline int get_cell_idx(size_t particle_idx, const float3* positions) {
        int x_idx = clamp<int>(int(((positions[particle_idx].x - BOX_MIN[0]) / CELL_SIZE)), 0, X_DIM - 1);
        int y_idx = clamp<int>(int(((positions[particle_idx].y - BOX_MIN[1]) / CELL_SIZE)), 0, Y_DIM - 1);
        int z_idx = clamp<int>(int(((positions[particle_idx].z - BOX_MIN[2]) / CELL_SIZE)), 0, Z_DIM - 1);
        return x_idx + (y_idx * X_DIM) + (z_idx * X_DIM * Y_DIM); 
    }

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

    __global__ void sort_particles_into_cells(
        float3* positions, 
        int* tmp, 
        int* cell_offsets,
        int* starts_LC
    );

    __global__ void update_cells(
        int* cells, 
        int* tmp, 
        int* cell_offsets, 
        int* starts_LC,
        float3* positions
    );

    __global__ void get_number_of_neighbors_LC_OPT(
        int* starts, 
        float3* positions, 
        int* starts_LC, 
        int* cells
    );
    
    __global__ void make_verlet_lists_LC_OPT(
        int* verletLists, 
        int* starts, 
        float3* positions, 
        int* starts_LC, 
        int* cells
    );
} // namespace ppb::cuda::nbody