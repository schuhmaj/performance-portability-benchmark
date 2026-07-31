#pragma once

#include "constants.cuh"

namespace ppb::cuda::nbody {
    //------------------------------------------------------------------HELPER FUNCTIONS---------------------------------------------------------------------
    __device__ inline float4 make_float4_add(const float4 a, const float4 b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.f);
    }

    __device__ inline float4 make_float4_sub(const float4 a, const float4 b) {
        return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.f);
    }

    __device__ inline float4 make_float4_scale(const float4 v, const float s) {
        return make_float4(v.x * s, v.y * s, v.z * s, 0.f);
    }

    __device__ inline float dot3(const float4 a, const float4 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    } 

    __device__ inline bool is_in_bounds(int idx, int offset) {
        int offset_idx = idx + offset;
        int x_idx = idx % x_dim;
        int y_idx = (idx / x_dim) % y_dim;
        int z_idx = (idx / (x_dim * y_dim));
        int x_offset = offset_idx % x_dim;
        int y_offset = (offset_idx / x_dim) % y_dim;
        int z_offset = (offset_idx / (x_dim * y_dim));

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


    __device__ inline int get_cell_idx(size_t particle_idx, const float4* positions) {
        int x_idx = clamp<int>(int(std::ceil((positions[particle_idx].x - boxMin[0]) / cell_size)), 0, x_dim - 1);
        int y_idx = clamp<int>(int(std::ceil((positions[particle_idx].y - boxMin[1]) / cell_size)), 0, y_dim - 1);
        int z_idx = clamp<int>(int(std::ceil((positions[particle_idx].z - boxMin[2]) / cell_size)), 0, z_dim - 1);
        return x_idx + (y_idx * x_dim) + (z_idx * x_dim * y_dim); 
    }

    __global__ void printStartsCells(int* starts, int* cells) {
        size_t numCells = x_dim * y_dim * z_dim;
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


    __global__ void update_cells(
        int* cells, 
        int* tmp, 
        int* cell_offsets, 
        int* starts,
        float4* cells_positions,
        float4* positions
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        } 
        
        size_t idx = starts[tmp[i]];
        size_t offset = cell_offsets[i];
        size_t position = idx + offset;
        cells[position] = i;
        cells_positions[position] = positions[i];
    }

    __global__ void update_positions(
        float4* positions, 
        const float4* velocities, 
        float4* forces, 
        float4* oldForces, 
        int* tmp, 
        int* cell_offsets,
        int* starts 
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        constexpr float mass = 1.0;
        const float4 force = forces[i];
        const float4 velocity = velocities[i];
        oldForces[i] = force;
        forces[i].x = globalForce[0];
        forces[i].y = globalForce[1];
        forces[i].z = globalForce[2];

        const float3 velocityPart = {velocity.x * deltaT, velocity.y * deltaT, velocity.z * deltaT};
        const float tt2m = deltaT * deltaT / (2.0f * mass);
        const float3 forcePart = {force.x * tt2m, force.y * tt2m, force.z * tt2m};
        const float3 displacement = {velocityPart.x + forcePart.x, velocityPart.y + forcePart.y, velocityPart.z + forcePart.z};
        positions[i] = {positions[i].x + displacement.x, positions[i].y + displacement.y, positions[i].z + displacement.z, 0.f};
        

        int idx = get_cell_idx(i, positions);
        int offset = atomicAdd(&starts[idx + 1], 1); //returns the value at starts[idx + 1] *before* adding 1.
        tmp[i] = idx;
        cell_offsets[i] = offset;
    }


    __global__ void update_velocities(
        float4* velocities, 
        const float4* forces, 
        const float4* oldForces
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        constexpr float mass = 1.0;
        const float4 force = forces[i];
        const float4 oldForce = oldForces[i];
        const float4 velocity = velocities[i];

        const float3 forcePart = {force.x + oldForce.x, force.y + oldForce.y, force.z + oldForce.z};
        const float t2m =  deltaT / (2.0f * mass);
        const float3 velChange = {forcePart.x * t2m, forcePart.y * t2m, forcePart.z * t2m};
        velocities[i] = {velocity.x + velChange.x, velocity.y + velChange.y, velocity.z + velChange.z, 0.f};
    }

    
    __global__ void compute_forces_old(
        const float4* __restrict__ positions,
        float4* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        float4 fi = make_float4(0.f, 0.f, 0.f, 0.f);
        size_t idx = get_cell_idx(i, positions);
        for (size_t offset = 0; offset < 27; offset++) {
            if (!is_in_bounds(idx, offsets[offset])) continue; 
            idx += offsets[offset];
            size_t start = starts[idx];
            size_t end = starts[idx + 1];
            for (size_t k = start; k < end; k++) {
                size_t j = cells[k];
                if (i >= j) continue; //N3L via natural ordering of indicies

                const float4 dr = make_float4_sub(positions[i], positions[j]);
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
                
                const float4 f = make_float4_scale(dr, fac);
                fi = make_float4_add(fi, f); 
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

    __global__ void compute_forces(
        const float4* __restrict__ cells_positions,
        float4* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        float4 fi = make_float4(0.f, 0.f, 0.f, 0.f);
        size_t idx = get_cell_idx(i, cells_positions);
        float4 pi = cells_positions[i];
        size_t ci = cells[i];
        for (size_t offset = 0; offset < 27; offset++) {
            if (!is_in_bounds(idx, offsets[offset])) continue; 
            idx += offsets[offset];
            size_t start = starts[idx];
            size_t end = starts[idx + 1];
            for (size_t k = start; k < end; k++) {
                size_t j = cells[k];
                float4 pj = cells_positions[k];
                if (ci >= j) continue; //N3L via natural ordering of indicies

                const float4 dr = make_float4_sub(pi, pj);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) >= cutoff_radius) continue; // = here too because less atomics
               
                const float sigma = 1.0f;
                const float sigmaSquared = sigma * sigma;
                const float epsilon24 = 24.0f; // 1.0 * 24.0

                const float invdr2 = 1.0f / dr2;
                float lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const float lj12 = lj6 * lj6;
                const float lj12m6 = lj12 - lj6;
                const float fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                
                const float4 f = make_float4_scale(dr, fac);
                fi = make_float4_add(fi, f); 
                atomicAdd(&forces[j].x, f.x * -1.0f);
                atomicAdd(&forces[j].y, f.y * -1.0f);
                atomicAdd(&forces[j].z, f.z * -1.0f);
            }
            idx -= offsets[offset];
        }

        atomicAdd(&forces[ci].x, fi.x);
        atomicAdd(&forces[ci].y, fi.y);
        atomicAdd(&forces[ci].z, fi.z);
    }


    __global__ void compute_forces_colored(
        int color,
        const float4* __restrict__ positions,
        float4* __restrict__ forces,
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
        if (!is_in_bounds(idx, 0)) return;
        int startBaseCell = starts[idx];
        int endBaseCell = starts[idx + 1];

        // Iterations: for each particle in base cell iterate through all the particles of the 8 cell region
        for (int q = startBaseCell; q < endBaseCell; q++) {
            float4 fi = make_float4(0.f, 0.f, 0.f, 0.f);
            int i = cells[q];
#pragma unroll 8
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
                    const float4 dr = make_float4_sub(positions[i], positions[j]);
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
                
                    const float4 f = make_float4_scale(dr, fac);
                    fi = make_float4_add(fi, f); 
                    forces[j] = make_float4_sub(forces[j], f);
                }
                idx -= offset;
            }
            forces[i] = make_float4_add(forces[i], fi);
        }

        //non-base-cell interactions
        for (int o = 0; o < 12; o+=2) {
            int cell_i = idx + offsets[offsets_colored_non_base_cell[o]];
            int cell_j = idx + offsets[offsets_colored_non_base_cell[o+1]];
            if (!is_in_bounds(cell_i, 0) || !is_in_bounds(cell_j, 0)) continue;
            int start_cell_i = starts[cell_i];     
            int end_cell_i = starts[cell_i + 1]; 
            int start_cell_j = starts[cell_j];
            int end_cell_j = starts[cell_j + 1]; 
            for (int i = start_cell_i; i < end_cell_i; i++) {
                int ci = cells[i];
                float4 fi = make_float4(0.f, 0.f, 0.f, 0.f);
                for (int j = start_cell_j; j < end_cell_j; j++) {
                    int cj = cells[j];
                    const float4 dr = make_float4_sub(positions[ci], positions[cj]);
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
                
                    const float4 f = make_float4_scale(dr, fac);
                    fi = make_float4_add(fi, f); 
                    forces[cj] = make_float4_sub(forces[cj], f);
                }
                forces[ci] = make_float4_add(forces[ci], fi);
            }
        }
    }

    /**
    * @param 'current_idx' is the current index inside of 'cells'
    * @param 'offset' is the offset which we have to the base_idx
    * @param 'base_cell_idx' is the index of the base cell
    */
    __device__ inline int2 get_next_element_neighborhood(int current_idx, int base_cell_idx, int current_offset, const int* starts) {
        //if next element still in current cell, return that
        if (current_idx != -1) {
            int end_current_cell = starts[base_cell_idx + offsets[current_offset] + 1];
            if (current_idx + 1 < end_current_cell) {
                return make_int2(current_idx + 1, current_offset);
            }
            current_offset++;
        }

        //if next element in different cell, go to next *non-empty* cell
        for (current_offset; current_offset < 27; current_offset++) {
            if (!is_in_bounds(base_cell_idx, offsets[current_offset])) continue;
            base_cell_idx += offsets[current_offset];
            int start_cell = starts[base_cell_idx];
            int end_cell = starts[base_cell_idx + 1]; 

            if (end_cell - start_cell == 0) {
                base_cell_idx -= offsets[current_offset];
                continue;
            }
            else return make_int2(start_cell, current_offset);
        }
       
        //if there there is no next element (i.e. current_idx is the last index of the neighborhood) return this
        return make_int2(INT_MAX, -1);
    }

    __device__ inline int get_num_neighbors(int base_cell_idx, const int* starts) {
        int num_neighbors = 0;
        for (int i = 0; i < 27; i++) {
            if (!is_in_bounds(base_cell_idx, offsets[i])) continue;
            base_cell_idx += offsets[i];
            num_neighbors += starts[base_cell_idx + 1] - starts[base_cell_idx];
            base_cell_idx -= offsets[i];
        }
        return num_neighbors;
    }

    __global__ void compute_forces_optimized(
        const float4* __restrict__ cells_positions,
        float4* __restrict__ forces,
        const int* __restrict__ starts,
        const int* __restrict__ cells,
        const int& shmem_size //size of shared memory in float3
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }
        extern __shared__ float3 shared_neighbors[];
        
        float4 pi = cells_positions[i];
        float4 fi = make_float4(0.f, 0.f, 0.f, 0.f);
        int idx = get_cell_idx(i, cells_positions);
        int shmem_tile_id = 0;

        //determine #cells in tile (= threadblock)
        if (i == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
            shared_neighbors[0].x = 0.f;
        }
        __syncthreads();
        if (i == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
            shmem_tile_id = (int)atomicAdd(&shared_neighbors[0].x, 1.f);
        }
        __syncthreads();
        int num_cells_in_tile = (int)(shared_neighbors[0].x);
        if (num_cells_in_tile > shmem_size) {
            printf("ERROR!\n");
            return;
        } //exit program if num_cells_in_tile > shmem_size. (might make this nicer in the future but probably not. Just tweak the tile size if need be.)
        
        //determine #neighbors the thread has to iterate over
        int num_neighbors = get_num_neighbors(idx, starts);

        //determine start + size of shmem region for each cell
        int size_shmem_tile = shmem_size / num_cells_in_tile; //for now idc about the remainder. That part is just gonna be unused. Will (probably) optimize this later.
        if (i == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
            shared_neighbors[shmem_tile_id].x = (float)idx;
        }
        __syncthreads();
        for (int t = 0; t < num_cells_in_tile; t++) {
            if (shared_neighbors[t].x == (float)idx) shmem_tile_id = t;
        }
        int start_shmem_tile = shmem_tile_id * size_shmem_tile; 
        //printf("Thread %u: start_shmem_tile: %lu\n", i, start_shmem_tile);

        int current_idx;
        int current_offset;
        for (int k = 0; k < num_neighbors; k += size_shmem_tile) {
            //load shmem region, for now only one thread per cell (in this case the first thread of the cell does all the copy work)
            //NOTE: THIS IS A BIG BOTTLENECK. WILL NOT BE CHANGED SINCE THIS OPTIMIZATION IS POINTLESS!
            if (i == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
                if (k == 0) {
                    current_idx = -1;
                    current_offset = 0;
                }
                for (int j = 0; j < size_shmem_tile; j++) { 
                    int2 idx_and_offset = get_next_element_neighborhood(current_idx, idx, current_offset, starts);
                    current_idx = idx_and_offset.x;
                    current_offset = idx_and_offset.y;
                    if (current_offset == -1) break; //we're at the end of the neighborhood. There is nothing left to copy.
                    float4 loaded_position = cells_positions[current_idx];
                    shared_neighbors[start_shmem_tile + j].x = loaded_position.x;
                    shared_neighbors[start_shmem_tile + j].y = loaded_position.y;
                    shared_neighbors[start_shmem_tile + j].z = loaded_position.z;
                }
            }
            __syncthreads(); 

            //force computation (NO N3L!! -> no shared memory bank conflicts (and no headache))
            int end_shmem_tile = k / size_shmem_tile == num_neighbors / size_shmem_tile
                ? start_shmem_tile + (num_neighbors - ((num_neighbors / size_shmem_tile) * size_shmem_tile))
                : start_shmem_tile + size_shmem_tile;
            for (int j = start_shmem_tile; j < end_shmem_tile; j++) {
                float4 pj = {shared_neighbors[j].x, shared_neighbors[j].y, shared_neighbors[j].z, 0.f};
                
                if (pi.x == pj.x && pi.y == pj.y && pi.z == pj.z) continue; //technically a bit meh cause two different particles could be on exactly the same position but whatever

                const float4 dr = make_float4_sub(pi, pj);
                const float dr2 = dot3(dr, dr);
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
                
                const float4 f = make_float4_scale(dr, fac);
                fi = make_float4_add(fi, f); 
            }
            __syncthreads();
        }
        forces[cells[i]] = make_float4_add(forces[cells[i]], fi);
    }
} // namespace ppb::cuda::nbody