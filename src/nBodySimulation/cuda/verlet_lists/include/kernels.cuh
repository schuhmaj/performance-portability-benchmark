#pragma once 

#include <cooperative_groups.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <float.h>
#include "constants.cuh"

namespace ppb::cuda::nbody {
    __device__ inline float3 make_float3_add(const float3 a, const float3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ inline float3 make_float3_sub(const float3 a, const float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ inline float3 make_float3_scale(const float3 v, const float s) {
        return make_float3(v.x * s, v.y * s, v.z * s);
    }

    __device__ inline float dot3(const float3 a, const float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    } 

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
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        constexpr float mass = 1.0;
        const float3 force = forces[i];
        const float3 velocity = velocities[i];
        oldForces[i] = force;
        forces[i].x = GLOBAL_FORCE[0];
        forces[i].y = GLOBAL_FORCE[1];
        forces[i].z = GLOBAL_FORCE[2];

        const float3 velocityPart = {velocity.x * DELTA_T, velocity.y * DELTA_T, velocity.z * DELTA_T};
        const float tt2m = DELTA_T * DELTA_T / (2.0f * mass);
        const float3 forcePart = {force.x * tt2m, force.y * tt2m, force.z * tt2m};
        const float3 displacement = {velocityPart.x + forcePart.x, velocityPart.y + forcePart.y, velocityPart.z + forcePart.z};
        positions[i] = {positions[i].x + displacement.x, positions[i].y + displacement.y, positions[i].z + displacement.z};
    }

    __global__ void update_velocities(float3* velocities, const float3* forces, const float3* oldForces) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        constexpr float mass = 1.0;
        const float3 force = forces[i];
        const float3 oldForce = oldForces[i];
        const float3 velocity = velocities[i];

        const float3 forcePart = {force.x + oldForce.x, force.y + oldForce.y, force.z + oldForce.z};
        const float t2m =  DELTA_T / (2.0f * mass);
        const float3 velChange = {forcePart.x * t2m, forcePart.y * t2m, forcePart.z * t2m};
        velocities[i] = {velocity.x + velChange.x, velocity.y + velChange.y, velocity.z + velChange.z};
    }

    __global__ void compute_forces(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ verletLists,
        const int* __restrict__ starts
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        float3 fi = make_float3(0.f, 0.f, 0.f);
        size_t start = starts[i];
        size_t end = starts[i + 1];
        for (size_t k = start; k < end; ++k) {
            size_t j = verletLists[k];
           
            if (i >= j) continue; //N3L via natural ordering of indicies

            const float sigma = 1.0f;
            const float sigmaSquared = sigma * sigma;
            const float epsilon24 = 24.0f; // 1.0 * 24.0

            const float3 dr = make_float3_sub(positions[i], positions[j]);
            const float dr2 = dot3(dr, dr);
            if (std::sqrt(dr2) >= CUTOFF_RADIUS) continue;

            const float invdr2 = 1.0f / dr2;
            float lj6 = sigmaSquared * invdr2;
            lj6 = lj6 * lj6 * lj6;
            const float lj12 = lj6 * lj6;
            const float lj12m6 = lj12 - lj6;
            const float fac = epsilon24 * (lj12 + lj12m6) * invdr2;

            const float3 f = make_float3_scale(dr, fac);
            fi = make_float3_add(fi, f);
            atomicAdd(&forces[j].x, f.x * -1.0f);
            atomicAdd(&forces[j].y, f.y * -1.0f);
            atomicAdd(&forces[j].z, f.z * -1.0f);
        }
        atomicAdd(&forces[i].x, fi.x);
        atomicAdd(&forces[i].y, fi.y);
        atomicAdd(&forces[i].z, fi.z);
    }

    __global__ void get_number_of_neighbors(int* starts, float3* positions) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        int neighbors = 0;
        float3 pi = positions[i];
        for (size_t j = 0; j < NUM_PARTICLES; j++) {
            if (i >= j) continue;
            const float3 dr = make_float3_sub(pi, positions[j]);
            const float dr2 = dot3(dr, dr);
            if (std::sqrt(dr2) <= CUTOFF_RADIUS + VERLET_SKIN) {
                neighbors++;
            }
        }

        starts[i + 1] = neighbors;
    }

    __global__ void make_verlet_lists(int* verletLists, int* starts, float3* positions) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        float3 pi = positions[i];
        size_t base = starts[i];
        int offset = 0;
        for (size_t j = 0; j < NUM_PARTICLES; j++) {
            if (i >= j) continue;
            const float3 dr = make_float3_sub(pi, positions[j]);
            const float dr2 = dot3(dr, dr);
            if (std::sqrt(dr2) <= CUTOFF_RADIUS + VERLET_SKIN) {
                verletLists[base + offset] = j;
                offset++;
            }
        }
    }

//--------------------------------------------- VERLET CLUSTER LISTS --------------------------------------------------
#ifdef PPB_ENABLE_VERLET_CLUSTER_LISTS
    __global__ void printStartsTowers(int* starts_towers, size_t num_towers) {
        printf("starts_towers:\n");
        for (int i = 0; i <= num_towers; i++) {
            printf("%d, ", starts_towers[i]);
        }
        printf("\n");
    }

    __global__ void printClusters(int* clusters, size_t size) {
        printf("clusters:\n");
        for (int i = 0; i < size; i++) {
            printf("%d, ", clusters[i]);
        }
    }

    __global__ void printBB(BoundingBox* BB, size_t num_boxes) {
        printf("bounding boxes:\n");
        for (int i = 0; i < num_boxes; i++) {
            float3 l = BB[i].lowerCorner;
            float3 u = BB[i].upperCorner;
            printf("BB[%d]:\nlower_corner: (%f, %f, %f)\nupper_corner: (%f, %f, %f)\n", i, l.x, l.y, l.z, u.x, u.y, u.z);
        }
    }
    
    __global__ void printPairList(int* starts, int num_clusters, int* cluster_pairs, int num_pairs) {
        printf("starts:\n");
        for (int i = 0; i < num_clusters + 1; i++) {
            printf("%d, ", starts[i]);
        }
        printf("pair list:\n");
        for (int i = 0; i < num_pairs; i++) {
            printf("%d, ", cluster_pairs[i]);
        }
        printf("\n");
    }

    __global__ void printPositionsInTowers(int* positions_in_tower, size_t size) {
        printf("positions_in_towers:\n");
        for (int i = 0; i < size; i++) {
            printf("%d, ", positions_in_tower[i]);
        }
        printf("\n");
    }

    __global__ void printZCoordinates(float* z_coordinates, size_t size) {
        printf("z_coordinates:\n");
        for (int i = 0; i < size; i++) {
            printf("%f, ", z_coordinates[i]);
        }
        printf("\n");
    }

    __device__ size_t get_tower_id(int particle_idx, float3* __restrict__ positions, float grid_size) {
        size_t x_dim = util::ceilDiv((BOX_MAX[0] - BOX_MIN[0]), grid_size);
        size_t y_dim = util::ceilDiv((BOX_MAX[1] - BOX_MIN[1]), grid_size);
        size_t tower_x = clamp<int>(int(((positions[particle_idx].x - BOX_MIN[0]) / grid_size)), 0, x_dim - 1);
        size_t tower_y = clamp<int>(int(((positions[particle_idx].y - BOX_MIN[1]) / grid_size)), 0, y_dim - 1);
        return tower_x + (tower_y * x_dim);
    }

/*     __device__ int2 get_tower_xy(int particle_idx, float3* __restrict__ positions, float grid_size) {
        int x_dim = util::ceilDiv((BOX_MAX[0] - BOX_MIN[0]), grid_size);
        int y_dim = util::ceilDiv((BOX_MAX[1] - BOX_MIN[1]), grid_size);
        int tower_x = clamp<int>(int(((positions[particle_idx].x - BOX_MIN[0]) / grid_size)), 0, x_dim - 1);
        int tower_y = clamp<int>(int(((positions[particle_idx].y - BOX_MIN[1]) / grid_size)), 0, y_dim - 1);
        return make_int2(tower_x, tower_y);
    } */

    __global__ void get_tower_id_per_particle(
        float3* __restrict__ positions,
        size_t* __restrict__ particle_tower_id,
        float grid_size
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }
        
        size_t tower = get_tower_id(i, positions, grid_size);
        particle_tower_id[i] = tower;
    }

    __global__ void count_particles_in_towers(
        float3* __restrict__ positions, 
        int* __restrict__ starts_towers,
        float grid_size
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        size_t tower = get_tower_id(i, positions, grid_size);
        atomicAdd(&starts_towers[tower + 1], 1);
    }

    __global__ void add_dummy_particles_to_towers(int* __restrict__ starts_towers, int cluster_size, int num_towers) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_towers) {
            return;
        }

        starts_towers[i + 1] += (cluster_size - (starts_towers[i + 1] % cluster_size)) % cluster_size;
    }

    __global__ void get_particle_position_in_tower(
        float3* __restrict__ positions, 
        int* __restrict__ clusters, 
        int* __restrict__ starts_towers,
        int* __restrict__ positions_in_tower,
        float grid_size
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }
 
        size_t tower = get_tower_id(i, positions, grid_size);
        int position_in_tower = atomicAdd(&clusters[starts_towers[tower]], 1);
        position_in_tower++; //because clusters is just -1, -1, ..., -1 at the start (for dummy particles)
        positions_in_tower[i] = starts_towers[tower] + position_in_tower;
        //I would prefer to just do a cooperative grid sync here and then just insert the particles here but cooperative grid sync is bugged (causes memory issue for some reason)
    }

    __global__ void insert_particles_into_towers(
        float3* __restrict__ positions, 
        int* __restrict__ clusters, 
        float* __restrict__ z_coordinates,
        int* __restrict__ starts_towers,
        int* __restrict__ positions_in_tower,
        float grid_size
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        clusters[positions_in_tower[i]] = i;
        z_coordinates[positions_in_tower[i]] = positions[i].z;        
    }


    __global__ void init_clusters_and_z_coordinates(
        int* __restrict__ clusters, 
        float* __restrict__ z_coordinates,
        int size_clusters
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size_clusters) {
            return;
        }

        clusters[i] = -1;
        z_coordinates[i] = FLT_MAX;
    }

    __global__ void compute_bounding_boxes(
        BoundingBox* __restrict__ BB, 
        int* __restrict__ clusters, 
        float3* __restrict__ positions,
        int cluster_size,
        int num_clusters
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_clusters) {
            return;
        }

        float3 min = make_float3(INFINITY, INFINITY, INFINITY);
        float3 max = make_float3(-INFINITY, -INFINITY, -INFINITY);
        
        for (int j = 0; j < cluster_size; j++) {
            if (clusters[(i * cluster_size) + j] == -1) { //handle dummy particles
                continue; //we going to just pretend that the dummy particles are contained inside the bounding box of the normal particles
            }
            float3 xyz_current = positions[clusters[(i * cluster_size) + j]];
            float x_current = xyz_current.x;
            float y_current = xyz_current.y;
            float z_current = xyz_current.z;
            if (x_current < min.x) min.x = x_current;
            if (y_current < min.y) min.y = y_current;
            if (z_current < min.z) min.z = z_current;
            if (x_current > max.x) max.x = x_current;
            if (y_current > max.y) max.y = y_current;
            if (z_current > max.z) max.z = z_current;
        }

        //If the cluster consists entirely of dummy particles put its bounding box at (INF, INF, INF)
        if (min.x == INFINITY && min.y == INFINITY && min.z == INFINITY && 
            max.x == -INFINITY && max.y == -INFINITY && max.z == -INFINITY) {
            BB[i].lowerCorner = min;
            BB[i].upperCorner = min;
        } else {
            BB[i].lowerCorner = min;
            BB[i].upperCorner = max;
        }
    }

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
    * It's basically like a zig-zag motion from the lower-left corner of the domain to the upper-right corner.
    * Note that if neighbor_x = tower_idx_x and neighbor_y = tower_idx_y this function will also return false! */
    __device__ inline bool isForwardNeighbor(int neighbor_x, int neighbor_y, int tower_idx_x, int tower_idx_y) {
        return neighbor_x > tower_idx_x || neighbor_y > tower_idx_y || (neighbor_x == tower_idx_x && neighbor_y == tower_idx_y);
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
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size_clusters / 8) { //1 Thread per i-cluster
            return;
        }

        int pair_idx = 0;
        int base_pair_idx = starts[i]; //assuming an inclusive scan was done on starts beforehand!
        
        float interactionLength = CUTOFF_RADIUS + VERLET_SKIN;
        float interactionLengthSqr = interactionLength * interactionLength;
        BoundingBox bbi = BBM[i];

        for (int j = 0; j < size_clusters; j+=4) {
            int j_cluster = j / 4;
            if (j_cluster <= 2 * i + 1) continue; //N3L + skip same cluster
            float boxDistSquared = BBdistanceSquared(bbi, BBN[j_cluster]);
            
            if (boxDistSquared <= interactionLengthSqr) {
                if (count) {
                    starts[i + 1]++;
                } else {           
                    //printf("Thread %u: Adding cluster with id %d to cluster_pairs at %d\n", i, j/4, base_pair_idx + pair_idx);         
                    cluster_pairs[base_pair_idx + pair_idx] = j_cluster;
                    pair_idx++;
                }
            }
        }
    }
   
    /**
        Immediately excludes entire towers based on their x and y coordinates. 
        A much better optimization however, would be somehow using Linked Cells in the pair search...
    */
/*     __global__ void cluster_pair_search_optimized(
        BoundingBox* __restrict__ BBM,
        BoundingBox* __restrict__ BBN,
        bool count,
        int* __restrict__ cluster_pairs, 
        int* __restrict__ starts,
        int* __restrict__ starts_towers,
        size_t* __restrict__ tower_ids,
        int* __restrict__ clusters,
        float3* __restrict__ positions,
        float grid_size,
        int num_towers,
        int size_clusters
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size_clusters / 8) { //1 Thread per i-cluster
            return;
        }
        int pair_idx = 0;
        int base_pair_idx = starts[i]; //assuming an inclusive scan was done on starts beforehand!
        int x_dim = util::ceilDiv((boxMax[0] - boxMin[0]), grid_size);
        int y_dim = util::ceilDiv((boxMax[1] - boxMin[1]), grid_size);
        //printf("x_dim: %d, y_dim: %d\n", x_dim, y_dim);

        size_t tower = get_tower_id(clusters[8 * i], positions, grid_size);
        size_t tower_x = tower % x_dim;
        size_t tower_y = tower / x_dim;

        int tower_start = get_start_tower(tower, tower_ids, num_towers);
        if (tower_start == -1) {
            printf("ERROR!\n");
            return; //THIS SHOULD NOT HAPPEN!!!
        }
        int cluster_z = (8 * i) - starts_towers[tower_start];
        
        float interactionLength = cutoff_radius + verlet_skin;
        float interactionLengthSqr = interactionLength * interactionLength;
        size_t min_tower_x = clamp<int>(int(((BBM[i].lowerCorner.x - interactionLength) - boxMin[0]) / grid_size), 0, x_dim - 1);
        size_t min_tower_y = clamp<int>(int(((BBM[i].lowerCorner.y - interactionLength) - boxMin[0]) / grid_size), 0, y_dim - 1);
        size_t max_tower_x = clamp<int>(int(((BBM[i].upperCorner.x + interactionLength) - boxMin[0]) / grid_size), 0, x_dim - 1);
        size_t max_tower_y = clamp<int>(int(((BBM[i].upperCorner.y + interactionLength) - boxMin[0]) / grid_size), 0, y_dim - 1);
        //printf("BBM[%u]: min_x: %lu, min_y: %lu, max_x: %lu, max_y: %lu\n", i, min_tower_x, min_tower_y, max_tower_x, max_tower_y);

        for (int t = 0; t < num_towers; t++) {
            size_t neighbor_tower = tower_ids[t];
            size_t neighbor_tower_x = neighbor_tower % x_dim;
            size_t neighbor_tower_y = neighbor_tower / x_dim;
            if (neighbor_tower_x < min_tower_x || neighbor_tower_x > max_tower_x || 
                neighbor_tower_y < min_tower_y || neighbor_tower_y > max_tower_y) {
                continue; //skip non-empty towers that are out of range
            }
            if (!isForwardNeighbor(neighbor_tower_x, neighbor_tower_y, tower_x, tower_y)) continue; //N3L
            //if neighbor_tower is same tower as tower_idx then be sure to skip the j-clusters before the i-cluster and also the 2 j-clusters contained in the i-cluster.
            int neighbor_tower_start = get_start_tower(neighbor_tower, tower_ids, num_towers);
            int start_neighbor_tower = starts_towers[neighbor_tower_start] + (neighbor_tower == tower ? (cluster_z + 8) : 0);
            int end_neighbor_tower = starts_towers[neighbor_tower_start + 1];
            //printf("Thread %u: neighbor_tower_start = %d, start_neighbor_tower = %d\n", i, neighbor_tower_start, start_neighbor_tower);
            for (int j = start_neighbor_tower; j < end_neighbor_tower; j+=4) {
                float boxDistSquared = BBdistanceSquared(BBM[i], BBN[j / 4]);
                if (boxDistSquared <= interactionLengthSqr) {
                    if (count) {
                        starts[i + 1]++;
                    } else {           
                        //printf("Thread %u: Adding cluster with id %d to cluster_pairs at %d\n", i, j/4, base_pair_idx + pair_idx);         
                        cluster_pairs[base_pair_idx + pair_idx] = (j / 4);
                        pair_idx++;
                    }
                }
            }
        }
    } */
    //-------------------------------------------------------------------------------------------------
    __device__ inline void compute_interaction(
        float3& i_particle,
        int i_particle_idx,
        float3& j_particle,
        int j_particle_idx,
        float3& fi,
        float3* __restrict__ forces
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const float3 dr = make_float3_sub(i_particle, j_particle);
        const float dr2 = dot3(dr, dr);

        if (std::sqrt(dr2) >= CUTOFF_RADIUS) return;

/*         if (i_particle_idx == 1 || j_particle_idx == 1) {
            printf("Thread %u: %d <-> %d", i, i_particle_idx, j_particle_idx);
        } */

        const float sigma = 1.0f;
        const float sigmaSquared = sigma * sigma;
        const float epsilon24 = 24.0f; // 1.0 * 24.0
        
        const float invdr2 = 1.0f / dr2;
        float lj6 = sigmaSquared * invdr2;
        lj6 = lj6 * lj6 * lj6;
        const float lj12 = lj6 * lj6;
        const float lj12m6 = lj12 - lj6;
        const float fac = epsilon24 * (lj12 + lj12m6) * invdr2;

        const float3 f = make_float3_scale(dr, fac);
        fi = make_float3_add(fi, f);
        atomicAdd(&forces[j_particle_idx].x, f.x * -1.0f);
        atomicAdd(&forces[j_particle_idx].y, f.y * -1.0f);
        atomicAdd(&forces[j_particle_idx].z, f.z * -1.0f);
    }

    __global__ void compute_force_cluster_lists(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ clusters,
        const int* __restrict__ cluster_pairs,
        const int* __restrict__ starts, //starts for cluster_pairs
        int size_clusters
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= 4 * size_clusters) {
            return;
        }

        int i_cluster = i / 32;
        int i_particle_idx = clusters[i / 4];
        if (i_particle_idx == -1) return; //skip dummy particles
        int start_neighbors = starts[i_cluster];
        int end_neighbors = starts[i_cluster + 1];
        float3 fi = make_float3(0.f, 0.f, 0.f);
        float3 i_particle = positions[i_particle_idx];
        
        //Compute interactions within the i-cluster
        for (int j = 0; j < 2; j++) {
            int j_particle_idx = clusters[i_cluster * 8 + (j * 4) + (i % 4)];
            if (j_particle_idx == -1) continue; //skip dummy particles
            if (i_particle_idx >= j_particle_idx) continue; //N3L within the same i-cluster
            float3 j_particle = positions[j_particle_idx];
            compute_interaction(i_particle, i_particle_idx, j_particle, j_particle_idx, fi, forces);
        }

        //N3L by construction. The pair list contains each pair of interacting clusters only once.
        for (int j = start_neighbors; j < end_neighbors; j++) {
            int j_cluster = cluster_pairs[j];
            int j_particle_idx = clusters[j_cluster * 4 + (i % 4)];
            if (j_particle_idx == -1) continue; //skip dummy particles
            float3 j_particle = positions[j_particle_idx];
            compute_interaction(i_particle, i_particle_idx, j_particle, j_particle_idx, fi, forces);
        }

        int MASK = 0xffffffff;
        float fi1_x = __shfl_down_sync(MASK, fi.x, 1);
        float fi1_y = __shfl_down_sync(MASK, fi.y, 1);
        float fi1_z = __shfl_down_sync(MASK, fi.z, 1);
        float fi2_x = __shfl_down_sync(MASK, fi.x, 2);
        float fi2_y = __shfl_down_sync(MASK, fi.y, 2);
        float fi2_z = __shfl_down_sync(MASK, fi.z, 2);
        float fi3_x = __shfl_down_sync(MASK, fi.x, 3);
        float fi3_y = __shfl_down_sync(MASK, fi.y, 3);
        float fi3_z = __shfl_down_sync(MASK, fi.z, 3);
        
        //We have 4 threads per i-particle. We assign the first of those threads to be the one that adds the accumulated fi's to the i-particle.
        if (i % 4 == 0) { 
            fi.x += fi1_x + fi2_x + fi3_x;
            fi.y += fi1_y + fi2_y + fi3_y;
            fi.z += fi1_z + fi2_z + fi3_z;
            atomicAdd(&forces[i_particle_idx].x, fi.x);
            atomicAdd(&forces[i_particle_idx].y, fi.y);
            atomicAdd(&forces[i_particle_idx].z, fi.z);
        }
    }

#elif PPB_ENABLE_VERLET_LISTS_LC_OPTIMIZATION
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
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        } 

        int idx = get_cell_idx(i, positions);
        int offset = atomicAdd(&starts_LC[idx + 1], 1); //returns the value at starts[idx + 1] *before* adding 1.
        tmp[i] = idx;
        cell_offsets[i] = offset;
    }

    __global__ void update_cells(
        int* cells, 
        int* tmp, 
        int* cell_offsets, 
        int* starts_LC,
        float3* positions
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        } 
        
        size_t idx = starts_LC[tmp[i]];
        size_t offset = cell_offsets[i];
        size_t position = idx + offset;
        cells[position] = i;
    }

    __global__ void get_number_of_neighbors_LC_OPT(int* starts, float3* positions, int* starts_LC, int* cells) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        int neighbors = 0;
        float3 pi = positions[i];
        int idx = get_cell_idx(i, positions);

        for (int o = 0; o < 27; o++) {
            if (!is_in_bounds(idx, o)) continue;
            idx += OFFSETS[o];
            int start = starts_LC[idx];
            int end = starts_LC[idx + 1];
            for (int k = start; k < end; k++) {
                int j = cells[k];
                if (i >= j) continue; 
                const float3 dr = make_float3_sub(pi, positions[j]);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) <= CUTOFF_RADIUS + VERLET_SKIN) {
                    neighbors++;
                }
            }
              
            idx -= OFFSETS[o];
        }

        starts[i + 1] = neighbors;
    }

    __global__ void make_verlet_lists_LC_OPT(int* verletLists, int* starts, float3* positions, int* starts_LC, int* cells) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        float3 pi = positions[i];
        int idx = get_cell_idx(i, positions);
        size_t base = starts[i];
        int offset = 0;
        
        for (int o = 0; o < 27; o++) {
            if (!is_in_bounds(idx, o)) continue;
            idx += OFFSETS[o];
            int start = starts_LC[idx];
            int end = starts_LC[idx + 1];
            for (int k = start; k < end; k++) {
                int j = cells[k];
                if (i >= j) continue; 
                const float3 dr = make_float3_sub(pi, positions[j]);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) <= CUTOFF_RADIUS + VERLET_SKIN) {
                    verletLists[base + offset] = j;
                    offset++;
                }
            }
            idx -= OFFSETS[o];
        }
    }

#endif
} // namespace ppb