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

    /**
    * @param 'result' stores the index inside of 'cells' where the next element lies.
    * @param 'base_idx' is the starting index inside of 'cells'
    * @param 'offset' is the offset which we have to the base_idx
    * @param 'base_cell_idx' is the index of the base cell
    * @param 'start_cell_offset' is the relative offset (0,...,7) of the cell 'base_idx' resides in
    */
    __device__ inline void get_next_element_neighbor_chunk(size_t& result, size_t base_idx, size_t offset, size_t base_cell_idx, int* starts, int* cells) {
        result = base_idx;
        for (size_t o = 0; o < 8; o++) {
            size_t offset_cell = offsets[offsets_colored[o]];
            if (!is_in_bounds(base_cell_idx, offset)) continue;
            base_cell_idx += offset_cell;
            size_t start_cell = starts[base_cell_idx];
            size_t end_cell = starts[base_cell_idx + 1];
            
            if (result + (end_cell - start_cell) < base_idx + offset) {
                result += (end_cell - start_cell);
                base_cell_idx -= offset;
                continue;
            }
            
            result += (offset - result);
        }
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

    __global__ void compute_forces_colored_optimized(
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
        }

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
        extern __shared__ float3 c08_block[];

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
    }

    //------------------------------------------- WORK IN PROGRESS ---------------------------------------------------
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