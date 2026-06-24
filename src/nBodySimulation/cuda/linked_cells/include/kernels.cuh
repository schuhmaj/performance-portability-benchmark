#pragma once

#include "constants.cuh"

namespace ppb {
    //------------------------------------------------------------------HELPER FUNCTIONS---------------------------------------------------------------------
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

    __device__ inline bool is_in_bounds(size_t idx, size_t offset) {
        size_t offset_idx = idx + offset;
        size_t x_idx = idx % x_dim;
        size_t y_idx = (idx / x_dim) % y_dim;
        size_t z_idx = (idx / (x_dim * y_dim));
        size_t x_offset = offset_idx % x_dim;
        size_t y_offset = (offset_idx / x_dim) % y_dim;
        size_t z_offset = (offset_idx / (x_dim * y_dim));

        if (std::abs((int)(x_idx - x_offset)) > 1) return false;
        else if (std::abs((int)(y_idx - y_offset)) > 1) return false;
        else if (std::abs((int)(z_idx - z_offset)) > 1) return false;
        else if (x_offset >= x_dim) return false;
        else if (y_offset >= y_dim) return false;
        else if (z_offset >= z_dim) return false;
        return true;
    }


    //taken from: https://github.com/dangets/cuda_examples/blob/master/clamp_function.cu (last accessed 14.6.26)
    template <typename T>
    __device__ inline T clamp(T val, T vMin, T vMax) {
        return min(max(val, vMin), vMax);
    }


    __device__ inline int get_cell_idx(size_t particle_idx, const float3* positions) {
        int x_idx = clamp<int>(int(std::ceil((positions[particle_idx].x - boxMin[0]) / cell_size)), 0, x_dim - 1);
        int y_idx = clamp<int>(int(std::ceil((positions[particle_idx].y - boxMin[1]) / cell_size)), 0, y_dim - 1);
        int z_idx = clamp<int>(int(std::ceil((positions[particle_idx].z - boxMin[2]) / cell_size)), 0, z_dim - 1);
        return x_idx + (y_idx * x_dim) + (z_idx * x_dim * y_dim); 
    } 


    __global__ void printStartsCells(int* starts, int* cells, size_t numCells, size_t numParticles) {
        printf("starts:\n");
        for (size_t j = 0; j <= numCells; j++) {
            printf("%d, ", starts[j]);
        }
        printf("\ncells:");
        for (size_t j = 0; j < numParticles; j++) {
            printf("%d, ", cells[j]);
        }
        printf("\n");
    }
    //-------------------------------------------------------------------------------------------------------------------------------------------------------


    __global__ void update_cells(int* cells, int* tmp, int* cell_offsets, int* starts) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        } 
        
        size_t idx = starts[tmp[i]];
        size_t offset = cell_offsets[i];
        size_t position = idx + offset;
        cells[position] = i;
    }

    __global__ void update_positions(
        float3* positions, 
        const float3* velocities, 
        float3* forces, 
        float3* oldForces, 
        int* tmp, 
        int* cell_offsets,
        int* starts 
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
        

        int idx = get_cell_idx(i, positions);
        int offset = atomicAdd(&starts[idx + 1], 1); //returns the value at starts[idx + 1] *before* adding 1.
        tmp[i] = idx;
        cell_offsets[i] = offset;
    }


    __global__ void update_velocities(
        float3* velocities, 
        const float3* forces, 
        const float3* oldForces
    ) {
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


    __global__ void compute_forces_colored(
        int color,
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        int* cells,
        int* starts
    ) {
        unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t num_cells = x_dim * y_dim * z_dim;
        if (t_id >= util::ceilDiv<size_t>(num_cells, 8)) {
            return;
        }

        // Map threads to their respective base cells based on the current color
        int x_dim_thread = x_dim % 2 != 0 && color & 1 == 0 ? util::ceilDiv<int>(x_dim, 2) : x_dim / 2;
        int y_dim_thread = y_dim % 2 != 0 && color & 2 == 0 ? util::ceilDiv<int>(y_dim, 2) : y_dim / 2;
        int z_dim_thread = z_dim % 2 != 0 && color & 4 == 0 ? util::ceilDiv<int>(z_dim, 2) : z_dim / 2;
        int x_thread = t_id % x_dim_thread;
        int y_thread = (t_id / x_dim_thread) % y_dim_thread;
        int z_thread = (t_id / (x_dim_thread * y_dim_thread));
        int x_cell = 2 * x_thread + (color % 2);
        int y_cell = 2 * y_thread + ((color>>1) % 2);
        int z_cell = 2 * z_thread + ((color>>2) % 2);
        int idx = x_cell + (y_cell * x_dim) + (z_cell * x_dim * y_dim);
        int startBaseCell = starts[idx];
        int endBaseCell = starts[idx + 1];

        // Iterations: for each particle in base cell iterate through all the particles of the 8 cell region
        for (int q = startBaseCell; q < endBaseCell; q++) {
            float3 fi = make_float3(0.f, 0.f, 0.f);
            int i = cells[q];
            for (int o = 0; o < 8; o++) {
                int offset = offsets[offsets_colored[o]];
                if (!is_in_bounds(idx, offset)) continue;
                idx += offset;
                int start = starts[idx];
                int end = starts[idx + 1];
                for (int k = start; k < end; k++) {
                    int j = cells[k];

                    //N3L via natural ordering of indicies (only necessary in same cell)
                    if (offset == 0 && i >= j) continue;
                    const float3 dr = make_float3_sub(positions[i], positions[j]);
                    const float dr2 = dot3(dr, dr); 

                    // = here too because this way we never get into a race condition with another cell of the same color
                    if (std::sqrt(dr2) >= cutoff_radius) continue;

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
                    forces[j] = make_float3_sub(forces[j], f);
                }
                idx -= offset;
            }
            forces[i] = make_float3_add(forces[i], fi);
        }
    }
    
    
    __global__ void compute_forces(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        float3 fi = make_float3(0.f, 0.f, 0.f);
        size_t idx = get_cell_idx(i, positions);
#pragma unroll 27
        for (size_t offset = 0; offset < 27; offset++) {
            if (!is_in_bounds(idx, offsets[offset])) continue; 
            idx += offsets[offset];
            size_t start = starts[idx];
            size_t end = starts[idx + 1];
            for (size_t k = start; k < end; k++) {
                size_t j = cells[k];
                if (i >= j) continue; //N3L via natural ordering of indicies

                const float3 dr = make_float3_sub(positions[i], positions[j]);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) >= cutoff_radius) continue; // = here too because less atomics in domain coloring

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
                atomicAdd(&forces[j].x, f.x * -1.0f);
                atomicAdd(&forces[j].y, f.y * -1.0f);
                atomicAdd(&forces[j].z, f.z * -1.0f);
            }
            idx -= offsets[offset];
        }

        atomicAdd(&forces[i].x, fi.x);
        atomicAdd(&forces[i].y, fi.y);
        atomicAdd(&forces[i].z, fi.z);
    }
} // namespace ppb