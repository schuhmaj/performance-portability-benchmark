#pragma once

#include "constants.cuh"

namespace ppb::cuda::nbody {
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

    __device__ inline bool is_in_bounds(int idx, int offset) {
        int offset_idx = idx + offset;
        int x_idx = idx % X_DIM;
        int y_idx = (idx / X_DIM) % Y_DIM;
        int z_idx = (idx / (X_DIM * Y_DIM));
        int x_offset = offset_idx % X_DIM;
        int y_offset = (offset_idx / X_DIM) % Y_DIM;
        int z_offset = (offset_idx / (X_DIM * Y_DIM));

        if (offset_idx < 0) return false;
        if (std::abs((int)(x_idx - x_offset)) > 1) return false;
        else if (std::abs((int)(y_idx - y_offset)) > 1) return false;
        else if (std::abs((int)(z_idx - z_offset)) > 1) return false;
        else if (x_offset >= X_DIM) return false;
        else if (y_offset >= Y_DIM) return false;
        else if (z_offset >= Z_DIM) return false;
        return true;
    }

    //taken from: https://github.com/dangets/cuda_examples/blob/master/clamp_function.cu (last accessed 14.6.26)
    template <typename T>
    __device__ inline T clamp(T val, T vMin, T vMax) {
        return min(max(val, vMin), vMax);
    }


    __device__ inline int get_cell_idx(size_t particle_idx, const float3* positions) {
        int x_idx = clamp<int>(int(((positions[particle_idx].x - BOX_MIN[0]) / CELL_SIZE)), 0, X_DIM - 1);
        int y_idx = clamp<int>(int(((positions[particle_idx].y - BOX_MIN[1]) / CELL_SIZE)), 0, Y_DIM - 1);
        int z_idx = clamp<int>(int(((positions[particle_idx].z - BOX_MIN[2]) / CELL_SIZE)), 0, Z_DIM - 1);
        return x_idx + (y_idx * X_DIM) + (z_idx * X_DIM * Y_DIM); 
    }

    __global__ void printStartsCells(int* starts, int* cells, float3* positions) {
        int num_cells = X_DIM * Y_DIM * Z_DIM;
        printf("starts:\n");
        for (int i = 0; i < num_cells; i++) {
            printf("%d, ", starts[i]);
        }
        printf("\ncells:\n");
        for (int i = 0; i < NUM_PARTICLES; i++) {
            printf("%d, ", cells[i]);
        }
        printf("\ncells_positions:\n");
        for (int i = 0; i < NUM_PARTICLES; i++) {
            float3 pi = positions[cells[i]];
            printf("%d (%f, %f, %f)\n", cells[i], pi.x, pi.y, pi.z);
        }
        printf("\n");
    }
    //-------------------------------------------------------------------------------------------------------------------------------------------------------


    __global__ void update_cells(
        int* cells, 
        int* tmp, 
        int* cell_offsets, 
        int* starts,
        float3* cells_positions,
        float3* positions
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        } 
        
        size_t idx = starts[tmp[i]];
        size_t offset = cell_offsets[i];
        size_t position = idx + offset;
        cells[position] = i;
        cells_positions[position] = positions[i];
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

    
    __global__ void compute_forces_unoptimized(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        float3 fi = make_float3(0.f, 0.f, 0.f);
        size_t idx = get_cell_idx(i, positions);
        for (size_t offset = 0; offset < 27; offset++) {
            if (!is_in_bounds(idx, OFFSETS[offset])) continue; 
            idx += OFFSETS[offset];
            size_t start = starts[idx];
            size_t end = starts[idx + 1];
            for (size_t k = start; k < end; k++) {
                size_t j = cells[k];
                if (i >= j) continue; //N3L via natural ordering of indicies

                const float3 dr = make_float3_sub(positions[i], positions[j]);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) >= CUTOFF_RADIUS) continue; // = here too because less atomics in domain coloring
                
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
            idx -= OFFSETS[offset];
        }

        atomicAdd(&forces[i].x, fi.x);
        atomicAdd(&forces[i].y, fi.y);
        atomicAdd(&forces[i].z, fi.z);
    }

    __global__ void compute_forces(
        const float3* __restrict__ cells_positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }

        float3 fi = make_float3(0.f, 0.f, 0.f);
        size_t idx = get_cell_idx(i, cells_positions);
        float3 pi = cells_positions[i];
        size_t ci = cells[i];
        for (size_t offset = 0; offset < 27; offset++) {
            if (!is_in_bounds(idx, OFFSETS[offset])) continue; 
            idx += OFFSETS[offset];
            size_t start = starts[idx];
            size_t end = starts[idx + 1];
            for (size_t k = start; k < end; k++) {
                size_t j = cells[k];
                float3 pj = cells_positions[k];
                if (ci >= j) continue; //N3L via natural ordering of indicies

                const float3 dr = make_float3_sub(pi, pj);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) >= CUTOFF_RADIUS) continue; // = here too because less atomics

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
            idx -= OFFSETS[offset];
        }

        atomicAdd(&forces[ci].x, fi.x);
        atomicAdd(&forces[ci].y, fi.y);
        atomicAdd(&forces[ci].z, fi.z);
    }


    __global__ void compute_forces_colored(
        int color,
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        int* cells,
        int* starts
    ) {
        unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (t_id >= NUM_CELLS_SAME_COLOR) {
            return;
        }

        // Map threads to their respective base cells based on the current color
        int x_dim_thread = X_DIM_NEAREST_4 / 2;
        int y_dim_thread = Y_DIM_NEAREST_4 / 2;
        int z_dim_thread = Z_DIM_NEAREST_4 / 2;
        int x_thread = t_id % x_dim_thread;
        int y_thread = (t_id / x_dim_thread) % y_dim_thread;
        int z_thread = (t_id / (x_dim_thread * y_dim_thread));
        int x_cell = 2 * x_thread + (color % 2);
        int y_cell = 2 * y_thread + ((color>>1) % 2);
        int z_cell = 2 * z_thread + ((color>>2) % 2);
        
        if (x_cell >= X_DIM || y_cell >= Y_DIM || z_cell >= Z_DIM) {
            return;
        }
 
        int idx = x_cell + (y_cell * X_DIM) + (z_cell * X_DIM * Y_DIM);
        int startBaseCell = starts[idx];
        int endBaseCell = starts[idx + 1];

        // Iterations: for each particle in base cell iterate through all the particles of the 8 cell region
        for (int q = startBaseCell; q < endBaseCell; q++) {
            float3 fi = make_float3(0.f, 0.f, 0.f);
            int i = cells[q];
#pragma unroll 8
            for (int o = 0; o < 8; o++) {
                int offset = OFFSETS[OFFSETS_COLORED[o]];
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
                    if (std::sqrt(dr2) >= CUTOFF_RADIUS) continue;

/*                 if (i == 0 || j == 0) {
                    printf("Thread %u: %d (Cell: %d) <-> %d (Cell: %d)\n", t_id, i, idx - offset, j, idx);
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
                    forces[j] = make_float3_sub(forces[j], f);
                }
                idx -= offset;
            }
            forces[i] = make_float3_add(forces[i], fi);
        }

        //non-base-cell interactions
        for (int o = 0; o < 12; o+=2) {
            if (!is_in_bounds(idx, OFFSETS[OFFSETS_COLORED_NON_BASE_CELL[o]]) 
            || !is_in_bounds(idx, OFFSETS[OFFSETS_COLORED_NON_BASE_CELL[o+1]])) continue;
            int cell_i = idx + OFFSETS[OFFSETS_COLORED_NON_BASE_CELL[o]];
            int cell_j = idx + OFFSETS[OFFSETS_COLORED_NON_BASE_CELL[o+1]];
            int start_cell_i = starts[cell_i];     
            int end_cell_i = starts[cell_i + 1]; 
            int start_cell_j = starts[cell_j];
            int end_cell_j = starts[cell_j + 1]; 
            for (int i = start_cell_i; i < end_cell_i; i++) {
                int ci = cells[i];
                float3 fi = make_float3(0.f, 0.f, 0.f);
                for (int j = start_cell_j; j < end_cell_j; j++) {
                    int cj = cells[j];
                    const float3 dr = make_float3_sub(positions[ci], positions[cj]);
                    const float dr2 = dot3(dr, dr); 

                    // = here too because this way we never get into a race condition with another cell of the same color
                    if (std::sqrt(dr2) >= CUTOFF_RADIUS) continue;
                
/*                     if (ci == 0 || cj == 0) {
                        printf("Thread %u: NON BASE CELL %d (Cell: %d) <-> %d (Cell: %d)\n", t_id, ci, cell_i, cj, cell_j);
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
                    forces[cj] = make_float3_sub(forces[cj], f);
                }
                forces[ci] = make_float3_add(forces[ci], fi);
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
            int end_current_cell = starts[base_cell_idx + OFFSETS[current_offset] + 1];
            if (current_idx + 1 < end_current_cell) {
                return make_int2(current_idx + 1, current_offset);
            }
            current_offset++;
        }

        //if next element in different cell, go to next *non-empty* cell
        for (; current_offset < 27; current_offset++) {
            if (!is_in_bounds(base_cell_idx, OFFSETS[current_offset])) continue;
            base_cell_idx += OFFSETS[current_offset];
            int start_cell = starts[base_cell_idx];
            int end_cell = starts[base_cell_idx + 1]; 

            if (end_cell - start_cell == 0) {
                base_cell_idx -= OFFSETS[current_offset];
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
            if (!is_in_bounds(base_cell_idx, OFFSETS[i])) continue;
            base_cell_idx += OFFSETS[i];
            num_neighbors += starts[base_cell_idx + 1] - starts[base_cell_idx];
            base_cell_idx -= OFFSETS[i];
        }
        return num_neighbors;
    }

    __global__ void compute_forces_optimized(
        const float3* __restrict__ cells_positions,
        float3* __restrict__ forces,
        const int* __restrict__ starts,
        const int* __restrict__ cells,
        const int& shmem_size //size of shared memory in float3
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= NUM_PARTICLES) {
            return;
        }
        extern __shared__ float3 shared_neighbors[];

        float3 pi = cells_positions[i];
        float3 fi = make_float3(0.f, 0.f, 0.f);
        int idx = get_cell_idx(i, cells_positions);
        int shmem_tile_id = 0;

        //determine #cells in tile (= threadblock)
        if (threadIdx.x == 0) {
            shared_neighbors[0].x = 0.f;
        }
        __syncthreads();
        if (threadIdx.x == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
            shmem_tile_id = (int)atomicAdd(&shared_neighbors[0].x, 1.f);
        }
        __syncthreads();
        int num_cells_in_tile = (int)(shared_neighbors[0].x);
        if (num_cells_in_tile > shmem_size) {
            printf("ERROR!\n");
            return;
        } //error if num_cells_in_tile > shmem_size. (might make this nicer in the future but probably not. Just tweak the tile size if need be.)
    
        //determine #neighbors the thread has to iterate over
        int num_neighbors = get_num_neighbors(idx, starts);

        //determine start + size of shmem region for each cell
        int size_shmem_tile = shmem_size / num_cells_in_tile; //for now I dont care about the remainder. That part is just gonna be unused.
        if (threadIdx.x == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
            shared_neighbors[shmem_tile_id].x = (float)idx;
        }
        __syncthreads();
        for (int t = 0; t < num_cells_in_tile; t++) {
            if (shared_neighbors[t].x == (float)idx) shmem_tile_id = t;
        }
        int start_shmem_tile = shmem_tile_id * size_shmem_tile; 

        int current_idx;
        int current_offset;
        for (int k = 0; k < num_neighbors; k += size_shmem_tile) {
            //load shmem region, for now only one thread per cell (in this case the first thread of the cell does all the copy work)
            //NOTE: THIS IS A BIG BOTTLENECK. WILL NOT BE CHANGED SINCE THIS OPTIMIZATION IS POINTLESS!
            if (threadIdx.x == 0 || get_cell_idx(i - 1, cells_positions) != idx) {
                if (k == 0) {
                    current_idx = -1;
                    current_offset = 0;
                }
                for (int j = 0; j < size_shmem_tile; j++) { 
                    int2 idx_and_offset = get_next_element_neighborhood(current_idx, idx, current_offset, starts);
                    current_idx = idx_and_offset.x;
                    current_offset = idx_and_offset.y;
                    if (current_offset == -1) break; //we're at the end of the neighborhood. There is nothing left to copy.
                    float3 loaded_position = cells_positions[current_idx];
                    shared_neighbors[start_shmem_tile + j].x = loaded_position.x;
                    shared_neighbors[start_shmem_tile + j].y = loaded_position.y;
                    shared_neighbors[start_shmem_tile + j].z = loaded_position.z;
                }
            }
            __syncthreads();

            //force computation (NO N3L!! -> no shared memory write bank conflicts (and no headache))
            int end_shmem_tile = k / size_shmem_tile == num_neighbors / size_shmem_tile
                ? start_shmem_tile + (num_neighbors - ((num_neighbors / size_shmem_tile) * size_shmem_tile))
                : start_shmem_tile + size_shmem_tile;
            for (int j = start_shmem_tile; j < end_shmem_tile; j++) {
                float3 pj = {shared_neighbors[j].x, shared_neighbors[j].y, shared_neighbors[j].z};
                
                if (pi.x == pj.x && pi.y == pj.y && pi.z == pj.z) continue; //technically a bit meh cause two different particles could be on exactly the same position but whatever

                const float3 dr = make_float3_sub(pi, pj);
                const float dr2 = dot3(dr, dr);
                if (std::sqrt(dr2) >= CUTOFF_RADIUS) continue;

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
            }
            __syncthreads();
        }
        forces[cells[i]] = make_float3_add(forces[cells[i]], fi);
    }
} // namespace ppb::cuda::nbody