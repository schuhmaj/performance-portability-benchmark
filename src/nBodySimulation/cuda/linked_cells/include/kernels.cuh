#pragma once

#include "constants.cuh"

namespace ppb {
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
        return 1;
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

    
    __global__ void compute_forces(
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
#pragma unroll 27
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
    }


    /**
    * @param 'base_idx' is the starting index inside of 'cells'
    * @param 'offset' is the offset which we have to the base_idx
    * @param 'base_cell_idx' is the index of the base cell
    */
    __device__ inline int get_next_element_neighborhood(int base_idx, int offset, int base_cell_idx, const int* starts, const int* cells) {
        int result = base_idx;
/* #pragma unroll 27 */ //not unrolled to preserve register space in compute_forces_optimized
        for (int i = 0; i < 27; i++) {
            if (!is_in_bounds(base_cell_idx, offsets[i])) continue;
            base_cell_idx += offsets[i];
            int start_cell = starts[base_cell_idx];
            int end_cell = starts[base_cell_idx + 1];
            
            if (result + (end_cell - start_cell) < base_idx + offset) {
                result += (end_cell - start_cell);
                base_cell_idx -= offsets[i];
                continue;
            } else {
                result += (offset - result);
                return result;
            }
        }
        return SIZE_MAX; //return this if offset starting at base_idx is no longer part of the neighborhood
    }

    __device__ inline int get_num_neighbors(int base_cell_idx, const int* starts) {
        int num_neighbors = 0;
#pragma unroll 27
        for (int i = 0; i < 27; i++) {
            if (!is_in_bounds(base_cell_idx, offsets[i])) continue;
            base_cell_idx += offsets[i];
            num_neighbors += starts[base_cell_idx + 1] - starts[base_cell_idx];
            base_cell_idx -= offsets[i];
        }
        return num_neighbors;
    }

    __global__ void compute_forces_optimized(
        const float4* __restrict__ positions,
        float4* __restrict__ forces,
        const float4* __restrict__ cells_positions,
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

        for (int k = 0; k < num_neighbors; k += size_shmem_tile) {
            //load shmem region, for now only one thread per cell (in this case the first thread of the cell does all the copy work)
            if (i == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
                for (int j = 0; j < size_shmem_tile; j++) {
                    int cell_idx = get_next_element_neighborhood(starts[idx], k + j, idx, starts, cells);
                    if (cell_idx == SIZE_MAX) break;
                    float4 loaded_position = cells_positions[cell_idx];
                    shared_neighbors[start_shmem_tile + j].x = loaded_position.x;
                    shared_neighbors[start_shmem_tile + j].y = loaded_position.y;
                    shared_neighbors[start_shmem_tile + j].z = loaded_position.z;
                }
            }
            __syncthreads(); 

/*         if (i == 0) {
            printf("shmem:\n");
            for (size_t j = 0; j < shmem_size; j++) {
                float3 pj = shared_neighbors[j];
                printf("x: %f, y: %f, z: %f\n", pj.x, pj.y, pj.z);
            }
        } */

            //force computation (NO N3L!! -> no shared memory bank conflicts (and no headache))
            int end_shmem_tile = k / size_shmem_tile == num_neighbors / size_shmem_tile
                ? start_shmem_tile + (num_neighbors - ((num_neighbors / size_shmem_tile) * size_shmem_tile))
                : start_shmem_tile + size_shmem_tile;
           // printf("Thread %u: end_shmem_tile: %lu\n", i, end_shmem_tile);
            for (int j = start_shmem_tile; j < end_shmem_tile; j++) {
                float4 pj = {shared_neighbors[j].x, shared_neighbors[j].y, shared_neighbors[j].z, 0.f};
                if (pi.x == pj.x && pi.y == pj.y && pi.z == pj.z) continue; //technically a bit meh cause two different particles could be on exactly the same position but whatever

                const float4 dr = make_float4_sub(pi, pj);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) >= cutoff_radius) continue;

                //printf("Thread %u: pi: (x: %f, y: %f, z: %f) <-> pj: %lu (x: %f, y: %f, z: %f)\n", i, pi.x, pi.y, pi.z, j, pj.x, pj.y, pj.z);
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
            forces[cells[i]] = make_float4_add(forces[cells[i]], fi);
            __syncthreads();
        }

    }
    //------------------------------------------- WORK IN PROGRESS (probably deprecated) ---------------------------------------------------
    /* __global__ void compute_forces_colored_optimized(
        int color,
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        int* cells,
        int* starts,
        size_t size_shmem //size of shmem in float3 units
    ) {
        size_t n_t = blockDim.x;
        unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t num_cells = x_dim * y_dim * z_dim;
        if (t_id >= util::ceilDiv<size_t>(num_cells, 8)) {
            return;
        }

        // Map threads to their respective base cells based on the current color
        size_t x_dim_thread = x_dim % 2 != 0 && color & 1 == 0 ? util::ceilDiv<int>(x_dim, 2) : x_dim / 2;
        size_t y_dim_thread = y_dim % 2 != 0 && color & 2 == 0 ? util::ceilDiv<int>(y_dim, 2) : y_dim / 2;
        size_t z_dim_thread = z_dim % 2 != 0 && color & 4 == 0 ? util::ceilDiv<int>(z_dim, 2) : z_dim / 2;
        size_t x_thread = threadIdx.x % x_dim_thread;
        size_t y_thread = (threadIdx.x / x_dim_thread) % y_dim_thread;
        size_t z_thread = (threadIdx.x / (x_dim_thread * y_dim_thread));
        size_t x_cell = 2 * x_thread + (color % 2);
        size_t y_cell = 2 * y_thread + ((color>>1) % 2);
        size_t z_cell = 2 * z_thread + ((color>>2) % 2);
        size_t idx = x_cell + (y_cell * x_dim) + (z_cell * x_dim * y_dim);
        size_t startBaseCell = starts[idx];
        size_t endBaseCell = starts[idx + 1];

        size_t total_num_neighbors;
        for (int o = 0; o < 8; o++) {
            int offset = offsets[offsets_colored[o]];
            if (!is_in_bounds(idx, offset)) continue;
            idx += offset;
            total_num_neighbors += (starts[idx + 1] - starts[idx]);
        } */

        /**
        Shared memory layout
        (one | means next entry in float3 array two || means here is where the next cell starts)
            ________________________________________________________________
            p00  | p01 | ... || p10 | p11 | ... || ... || ... | ... | ... ||  ...
            ----------------------------------------------------------------
            ________________________________________________________________ 
        ... f00  | f01 | ... || f10 | f11 | ... || ... || ... | ... | ... || 
            ----------------------------------------------------------------
        where pij indicates that this is the float3 of the position of the j-th particle of the i-th cell
        and   fij indicates that this is the float3 of the force of the j-th particle of the i-th cell

        Shared memory will contain as many positions and forces of the base cell as possible. If a base cell cannot
        be loaded completely the rest of it will always be loaded from global memory.
        */
        /* extern __shared__ float3 c08_block[];

        // Load shared memory. For now a single thread loads all the data. Obviously this can be optimized more.
        size_t shmem_idx = 0;
        for (size_t c = 0; c < 2; c++) {
            for (size_t i = startBaseCell; i < endBaseCell; i++, shmem_idx++) {
                if (shmem_idx >= size_shmem) goto SYNC;
                //thread 0 of each thread block is responsible for loading the entire base cell into shared memory
                if (threadIdx.x == 0) { 
                    size_t j = cells[i];
                    if (c == 0) c08_block[shmem_idx] = positions[j];
                    else c08_block[shmem_idx] = forces[j]; 
                } 
            } 
        }
        SYNC: __syncthreads();

        size_t numParticlesBaseCell = endBaseCell - startBaseCell;
        size_t startShmemBaseCell = threadIdx.x * (numParticlesBaseCell / n_t) + min(numParticlesBaseCell % n_t, (size_t)threadIdx.x);
        size_t endShmemBaseCell = threadIdx.x < n_t - 1 
            ? (threadIdx.x + 1) * (numParticlesBaseCell / n_t) + min(numParticlesBaseCell % n_t, (size_t)threadIdx.x) 
            : numParticlesBaseCell;
        
        // Iterate over the n_t neighbor-chunks
        for (size_t c = 0; c < n_t; c++) { 
            size_t size_chunk = total_num_neighbors % n_t != 0 && c < (total_num_neighbors % n_t) 
                ? (total_num_neighbors / n_t) + 1 
                : total_num_neighbors / n_t;
            for (size_t i = startShmemBaseCell; i < endShmemBaseCell; i++) { 
                float3 fi = make_float3(0.f, 0.f, 0.f);

                //get starting index of chunk
                size_t start_chunk = 0;
                size_t absolute_chunk_idx = (threadIdx.x + c) % n_t;
                size_t chunk_start_idx = absolute_chunk_idx * (total_num_neighbors / n_t) + min(total_num_neighbors % n_t, absolute_chunk_idx); 
                get_next_element_neighbor_chunk(start_chunk, 0, chunk_start_idx, idx, starts, cells);

                //iterate over neighbor-chunk and update forces between base-cell-chunk and neighbor-chunk
                for (size_t k = 0; k < size_chunk; k++) {
                    // get to next element in neighbor chunk
                    size_t j = 0;
                    get_next_element_neighbor_chunk(j, 0, start_chunk + k, idx, starts, cells); 

                    float3 dr = make_float3(0.f, 0.f, 0.f);
                    if (i < size_shmem) {
                        dr = make_float3_sub(c08_block[i], positions[j]);
                    } else {
                        dr = make_float3_sub(positions[i], positions[j]);
                    }

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
                if (i + numParticlesBaseCell < size_shmem) { 
                    c08_block[i + numParticlesBaseCell] = make_float3_add(c08_block[i + numParticlesBaseCell], fi);
                } else {
                    forces[i] = make_float3_add(forces[i], fi);
                }
            }
            // sync threads
            __syncthreads();
        } 

        // update forces in base cell. 
        // For now no N3L, because this way we can use the advantage of having multiple threads without synchronization overhead.
        // To be confirmed via benchmarking but I feel like since one cell doesn't contain that many particles in general this is actually the more performant solution (in general).
        for (size_t i = startShmemBaseCell; i < endShmemBaseCell; i++) {
            float3 acc = make_float3(0.f, 0.f, 0.f);
            float3 pi = i < size_shmem ? c08_block[i] : positions[i];
            float3 fi = i + numParticlesBaseCell < size_shmem ? c08_block[i + numParticlesBaseCell] : forces[i];
            for (size_t j = 0; j < numParticlesBaseCell * 2; j++) {
                float3 pj = j < size_shmem ? c08_block[j] : positions[j];
                float3 fj = j + numParticlesBaseCell < size_shmem ? c08_block[j + numParticlesBaseCell] : forces[j];
                
                const float3 dr = make_float3_sub(pi, pj);
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
                acc = make_float3_add(acc, f); 
                if (j + numParticlesBaseCell < size_shmem) {
                    c08_block[j + numParticlesBaseCell] = make_float3_sub(fj, f);
                } else {
                    forces[j] = make_float3_sub(fj, f);
                }
            }
            if (i + numParticlesBaseCell < size_shmem) {
                c08_block[i + numParticlesBaseCell] = make_float3_add(fi, acc);
            } else {
                forces[i] = make_float3_add(fi, acc);
            }
        }
    

        // Write back shared memory contents into global memory. For now this is just done by one single thread
        shmem_idx = 0;
        for (size_t c = 0; c < 2; c++) {
            for (size_t i = startBaseCell; i < endBaseCell; i++, shmem_idx++) {
                if (shmem_idx >= size_shmem) return;
                //thread 0 of each thread block is responsible for loading the entire base cell into shared memory
                if (threadIdx.x == 0) { 
                    size_t j = cells[i];
                    if (c == 0) continue;
                    else forces[j] = c08_block[shmem_idx]; 
                } 
            } 
        }
    } */

   /*  __device__ load_shmem_parallel() {
        
        // Load neighborhood into shared memory
        size_t total_num_neighbors;
        size_t size_shmem;   //size of the chunk that is loaded by the thread
        size_t start_shmem;  //where the chunk that is loaded by the thread starts in shared memory
     
        // init total_num_neighbors

        // init size_shmem
        if (total_num_neighbors % n_t != 0) {
            size_shmem = t_id < total_num_neighbors % n_t ? total_num_neighbors / n_t + 1 : total_num_neighbors;
        } else {
            size_shmem = total_num_neighbors / n_t;
        }

        // init start_shmem



        size_t at_cells = 0;
        size_t at_shmem = 0;
        for (size_t o = 0; o < 8; o++) {
            int offset = offsets[offsets_colored[o]];
            if (!is_in_bounds(idx, offset)) continue;
            idx += offset;
            int start_cell = starts[idx];
            int end_cell = starts[idx + 1];
            
            //get to starting index of chunk the thread must load into shared memory
            if (at_shmem == 0 && at_cells + (end_cell - start_cell) < start_shmem) {
                at_cells += (end_cell - start_cell);
                idx -= offset;
                continue;
            }
            
            //now the thread arrived at the starting index of the chunk it must load into shared memory
            size_t start_copy = at_shmem == 0 ? start_cell + (start_shmem - at_cells) : start_cell;
            size_t end_copy = at_shmem + ((end_cell - start_copy) * 2) >= size_shmem ? start_copy + ((size_shmem - at_shmem) / 2) : end_cell;
            for (size_t i = start_copy; i < end_copy; i++, at_shmem+=2) {
                c08_block[startShmem + at_shmem] = positions[i];
                c08_block[startShmem + at_shmem + 1] = forces[i];
            }
        }

        // Wait until entire neighborhood has been loaded into shared memory
        __syncthreads();
    } */
    
    
} // namespace ppb