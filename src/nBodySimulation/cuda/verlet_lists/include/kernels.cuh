#pragma once 

#include <cooperative_groups.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "constants.cuh"

namespace ppb {
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

    __global__ void update_positions(
        float3* positions, 
        const float3* velocities, 
        float3* forces, 
        float3* oldForces 
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        constexpr float mass = 1.0;
        const float3 force = forces[i];
        const float3 velocity = velocities[i];
        oldForces[i] = force;
        forces[i].x = globalForce[0];
        forces[i].y = globalForce[1];
        forces[i].z = globalForce[2];

        const float3 velocityPart = {velocity.x * deltaT, velocity.y * deltaT, velocity.z * deltaT};
        const float tt2m = deltaT * deltaT / (2.0f * mass);
        const float3 forcePart = {force.x * tt2m, force.y * tt2m, force.z * tt2m};
        const float3 displacement = {velocityPart.x + forcePart.x, velocityPart.y + forcePart.y, velocityPart.z + forcePart.z};
        positions[i] = {positions[i].x + displacement.x, positions[i].y + displacement.y, positions[i].z + displacement.z};
    }

    __global__ void update_velocities(float3* velocities, const float3* forces, const float3* oldForces) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        constexpr float mass = 1.0;
        const float3 force = forces[i];
        const float3 oldForce = oldForces[i];
        const float3 velocity = velocities[i];

        const float3 forcePart = {force.x + oldForce.x, force.y + oldForce.y, force.z + oldForce.z};
        const float t2m =  deltaT / (2.0f * mass);
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
        if (i >= numParticles) {
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
        if (i >= numParticles) {
            return;
        }

        int neighbors = 0;
        float3 pi = positions[i];
        for (size_t j = 0; j < numParticles; j++) {
            if (i == j) continue;
            const float3 dr = make_float3_sub(pi, positions[j]);
            const float dr2 = dot3(dr, dr);
            if (std::sqrt(dr2) <= cutoff_radius + verlet_skin) {
                neighbors++;
            }
        }

        starts[i + 1] = neighbors;
    }

    __global__ void make_verlet_lists(int* verletLists, int* starts, float3* positions) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        float3 pi = positions[i];
        size_t base = starts[i];
        int offset = 0;
        for (size_t j = 0; j < numParticles; j++) {
            if (i == j) continue;
            const float3 dr = make_float3_sub(pi, positions[j]);
            const float dr2 = dot3(dr, dr);
            if (std::sqrt(dr2) <= cutoff_radius + verlet_skin) {
                verletLists[base + offset] = j;
                offset++;
            }
        }
    }

//--------------------------------------------- VERLET CLUSTER LISTS --------------------------------------------------
#ifdef PPB_ENABLE_VERLET_CLUSTER_LISTS
    //taken from: https://github.com/dangets/cuda_examples/blob/master/clamp_function.cu (last accessed 14.6.26)
    template <typename T>
    __device__ inline T clamp(T val, T vMin, T vMax) {
        return min(max(val, vMin), vMax);
    }

    __device__ int get_tower_id(int particle_idx, float3* __restrict__ positions, float grid_size) {
        int x_dim = util::ceilDiv((boxMax[0] - boxMin[0]), grid_size);
        int y_dim = util::ceilDiv((boxMax[1] - boxMin[1]), grid_size);
        int tower_x = clamp<int>(int(std::ceil((positions[particle_idx].x - boxMin[0]) / grid_size)), 0, x_dim - 1);
        int tower_y = clamp<int>(int(std::ceil((positions[particle_idx].y - boxMin[1]) / grid_size)), 0, y_dim - 1);
        return tower_x + (tower_y * x_dim);
    }

    __device__ int2 get_tower_xy(int particle_idx, positions, grid_size) {
        int x_dim = util::ceilDiv((boxMax[0] - boxMin[0]), grid_size);
        int y_dim = util::ceilDiv((boxMax[1] - boxMin[1]), grid_size);
        int tower_x = clamp<int>(int(std::ceil((positions[particle_idx].x - boxMin[0]) / grid_size)), 0, x_dim - 1);
        int tower_y = clamp<int>(int(std::ceil((positions[particle_idx].y - boxMin[1]) / grid_size)), 0, y_dim - 1);
        return make_int2(tower_x, tower_y);
    }

    __global__ void count_elements_in_towers(
        float3* __restrict__ positions,
        int* __restrict__ starts_towers,
        float grid_size
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        int tower = get_tower_id(positions, grid_size);
        atomicAdd(starts_towers[tower], 1);
    }

    __global__ void add_dummy_particles_to_towers(int* __restrict__ starts_towers, int cluster_size) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= sizeof(starts_towers) - 1) {
            return;
        }

        starts_towers[i] += (cluster_size - (starts_towers[i] % cluster_size)) % cluster_size;
    }

    __global__ void insert_particles_into_towers(
        float3* __restrict__ positions, 
        int* __restrict__ clusters, 
        int* __restrict__ starts_towers,
        float grid_size
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }
 
        int tower = get_tower_id(i, positions, grid_size);
        int position_in_tower = atomicAdd(clusters[starts_towers[tower]], 1);
        position_in_tower++; //because clusters is just -1, -1, ..., -1 at the start (for dummy particles)
        cooperative_groups::this_grid.sync();
        clusters[starts_towers[tower] + position_in_tower] = i;
    }

    __global__ void sort_particles_along_z_axis(
        int* __restrict__ clusters, 
        int* __restrict__ starts_towers, 
        float3* __restrict__ positions
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= sizeof(starts_towers) - 1) {
            return;
        }

        int start = starts_towers[i];
        int size = starts_towers[i + 1] - start;
        thrust::sort(thrust::device, clusters + start, clusters + start + size, 
            [positions](int a, int b) int {
                // dummy particles should be part of the very last cluster
                if (a == -1) return true;
                if (b == -1) return false;
                // ordering between normal (non-dummy) particles
                return positions[a].z < positions[b].z;
            });
    } 

    __global__ void compute_bounding_boxes(BoundingBox* BB, int* __restrict__ clusters, int cluster_size) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= sizeof(clusters) / cluster_size) {
            return;
        }

        float3* min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3* max = make_float3(FLT_MIN, FLT_MIN, FLT_MIN);
        
        for (int j = 0; j < cluster_size; j++) {
            if (clusters[(i * cluster_size) + j] == -1) { //handle dummy particles
                continue; //we going to just pretend that the dummy particles are contained inside the bounding box of the normal particles
            }
            float x_current = clusters[(i * cluster_size) + j].x;
            float y_current = clusters[(i * cluster_size) + j].y;
            float z_current = clusters[(i * cluster_size) + j].z;
            if (x_current < min.x) min.x = x_current;
            if (y_current < min.y) min.y = y_current;
            if (z_current < min.z) min.z = z_current;
            if (x_current > max.x) max.x = x_current;
            if (y_current > max.y) max.y = y_current;
            if (z_current > max.z) max.z = z_current;
        }

        //if a cluster has lower corner = FLT_MAX, FLT_MAX, FLT_MAX and upper corner FLT_MIN, FLT_MIN, FLT_MIN it's a cluster
        //that contains only dummy particles.
        BB[i].lowerCorner = min;
        BB[i].upperCorner = max;
    }

    //Inspired by AutoPas: 
    //https://github.com/AutoPas/AutoPas/blob/master/src/autopas/utils/ArrayMath.h
    //https://github.com/AutoPas/AutoPas/blob/af9a1530fca6543aa651600751256cf408deaf13/src/autopas/containers/verletClusterLists/VerletClusterListsRebuilder.h
    __device__ inline float3 maxF3(float3& a, float3& b) {
        float3 result = make_float3(0.f, 0.f, 0.f);
#pragma unroll 3
        for (short i = 0; i < 3; i++) {
            result = max(a[i], b[i]);
        }
        return result;
    }

    __device__ inline float BBdistanceSquared(BoundingBox& a, BoundingBox& b) {
        float3 aToB = maxF3(make_float3(0.f, 0.f, 0.f), make_float3_sub(a.lower, b.upper));
        float3 bToA = maxF3(make_float3(0.f, 0.f, 0.f), make_float3_sub(b.lower, a.upper));
        return dot3(aToB) + dot3(bToA);
    }

    /** Order of N3L: iterate over all x from 0 to x_dim - 1 in one y-dimension, then repeat on the next *higher* y-dimension
    * It's basically like a zig-zag motion from the lower-left corner of the domain to the upper-right corner.
    * Note that if neighbor_x = tower_idx_x and neighbor_y = tower_idx_y this function will also return false! */
    __device__ inline bool isForwardNeighbor(int neighbor_x, int neighbor_y, int tower_idx_x, int tower_idx_y) {
        return neighbor_x > tower_idx_x || neighbor_y > tower_idx_y;
    }
    
    __global__ void cluster_pair_search(
        BoundingBox* __restrict__ BBM,
        BoundingBox* __restrict__ BBN,
        bool count,
        int* __restrict__ cluster_pairs, 
        int* __restrict__ starts,
        int* __restrict__ starts_towers,
        int* __restrict__ clusters,
        float3* __restrict__ positions,
        float grid_size
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (sizeof(BBM) / sizeof(BoundingBox))) {
            return;
        }
        if (!count) {
            int pair_idx = 0;
            int base_pair_idx = starts[i]; //assuming an inclusive scan was done on starts beforehand!
        }
        int x_dim = util::ceilDiv((boxMax[0] - boxMin[0]), grid_size);
        int y_dim = util::ceilDiv((boxMax[1] - boxMin[1]), grid_size);

        int2 tower_idx_xy = get_tower_xy(clusters[i], positions, grid_size);
        int tower_idx_x = tower_idx_xy.x;
        int tower_idx_y = tower_idx_xy.y;
        int tower_idx = tower_idx_x + (tower_idx_y * x_dim);
        int cluster_z = (8 * i) - starts_towers[tower_idx];
        
        float interactionLength = cutoff_radius + verlet_skin;
        float interactionLengthSqr = interactionLength * interactionLength;
        int min_neighbor_tower_x = clamp<int>(int((BBM[i].lowerCorner.x - interactionLength) / grid_size), 0, x_dim - 1);
        int min_neighbor_tower_y = clamp<int>(int((BBM[i].lowerCorner.y - interactionLength) / grid_size), 0, y_dim - 1);
        int max_neighbor_tower_x = clamp<int>(int((BBM[i].upperCorner.x + interactionLength) / grid_size), 0, x_dim - 1);
        int max_neighbor_tower_y = clamp<int>(int((BBM[i].upperCorner.y + interactionLength) / grid_size), 0, y_dim - 1);

        for (int neighbor_x = min_neighbor_tower_x; neighbor_x < max_neighbor_tower_x; neighbor_x++) {
            for (int neighbor_y = min_neighbor_tower_y; neighbor_y < max_neighbor_tower_y; neighbor_y++) {
                if (!isForwardNeighbor(neighbor_x, neighbor_y, tower_idx_x, tower_idx_y)) continue; //N3L
                //if neighbor_tower is same tower as tower_idx then be sure to skip the j-clusters before the i-cluster and also the 2 j-clusters contained in the i-cluster.
                int neighbor_tower_idx = neighbor_x + (neighbor_y * x_dim);
                int start_neighbor_tower = starts_towers[neighbor_tower_idx] + (neighbor_tower_idx == tower_idx ? (cluster_z + 8) : 0);
                int end_neighbor_tower = starts_towers[neighbor_tower_idx + 1];
                for (int j = start_neighbor_tower; j < end_neighbor_tower; j+=4) {
                    float boxDistSquared = boxDistSquared(BBM[i], BBN[j / 4]);
                    if (boxDistSquared <= interactionLengthSqr) {
                        if (count) {
                            starts[i + 1]++;
                        } else {                     
                            cluster_pairs[base_pair_idx + pair_idx] = (j / 4);
                            pair_idx++;
                        }
                    }
                }
            }
        }
    }
    //-------------------------------------------------------------------------------------------------
    __device__ inline void compute_interaction(
        int i_particle,
        int j_particle,
        float3& fi,
        const float3* __restrict__ positions,
        float3* __restrict__ forces
    ) {
        const float3 dr = make_float3_sub(positions[i_particle], positions[j_particle]);
        const float dr2 = dot3(dr, dr);
        if (dr2 >= cutoff_radius) continue;
            
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
        atomicAdd(&forces[j_particle].x, f.x * -1.0f);
        atomicAdd(&forces[j_particle].y, f.y * -1.0f);
        atomicAdd(&forces[j_particle].z, f.z * -1.0f);
    }

    __global__ void compute_force_cluster_lists(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ clusters,
        const int* __restrict__ cluster_pairs,
        const int* __restrict__ starts //starts for cluster_pairs
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= 4 * numParticles) {
            return;
        }

        int i_cluster = i / 32;
        int i_particle = i / 4;
        int start_neighbors = starts[i_cluster];
        int end_neighbors = starts[i_cluster + 1];
        float3 fi = make_float3(0.f, 0.f, 0.f);
        
        //Compute interactions within the i-cluster
        for (int j = 0; j < 2; j++) {
            int j_particle = clusters[i_cluster * 8 + (j * 4)] + (i % 4);
            if (i_particle >= j_particle) continue; //N3L within the same i-cluster
            compute_interaction(i_particle, j_particle, fi, positions, forces);
        }

        //N3L by construction. The pair list contains each pair of interacting clusters only once.
        for (int j_cluster = start_neighbors; j_cluster < end_neighbors; j_cluster++) {
            int j_particle = clusters[j_cluster * 4] + (i % 4);
            compute_interaction(i_particle, j_particle, fi, positions, forces);
        }

        //We have 4 threads per i-particle. We assign the first of those threads to be the one that adds the accumulated fi's to the i-particle.
        if (i % 4 == 0) { 
            int MASK = 0x1111<<(i % 32);
            float3 fi1 = __shfl_down_sync(MASK, fi, 1);
            float3 fi2 = __shfl_down_sync(MASK, fi, 2);
            float3 fi3 = __shfl_down_sync(MASK, fi, 3);
            fi = make_float3_add(fi, fi1);
            fi = make_float3_add(fi, fi2);
            fi = make_float3_add(fi, fi3);
            atomicAdd(&forces[i].x, fi.x);
            atomicAdd(&forces[i].y, fi.y);
            atomicAdd(&forces[i].z, fi.z);
        }
    }
#endif
} // namespace ppb