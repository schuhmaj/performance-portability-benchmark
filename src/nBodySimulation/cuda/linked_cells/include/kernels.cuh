#pragma once

#include "common/cuda/Cuda_Float3_Arithmetic.cuh"
#include "constants.cuh"

namespace ppb::cuda::nbody {
//------------------------------------------------------------------HELPER FUNCTIONS---------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------

    __global__ void update_cells(
        int* cells, 
        int* tmp, 
        int* cell_offsets, 
        int* starts,
        float3* cells_positions,
        float3* positions
    );

    __global__ void update_positions(
        float3* positions, 
        const float3* velocities, 
        float3* forces, 
        float3* oldForces, 
        int* tmp, 
        int* cell_offsets,
        int* starts 
    );

    __global__ void update_velocities(
        float3* velocities, 
        const float3* forces, 
        const float3* oldForces
    );
    
    __global__ void compute_forces_unoptimized(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    );

    __global__ void compute_forces(
        const float3* __restrict__ cells_positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    );

    __global__ void compute_forces_colored(
        int color,
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        int* cells,
        int* starts
    );

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
            if (!is_in_bounds(base_cell_idx, current_offset)) continue;
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
            if (!is_in_bounds(base_cell_idx, i)) continue;
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
        const int shmem_size //size of shared memory in float3
    );

    __global__ void compute_forces_optimized_alt(
        const float3* __restrict__ cells_positions,
        float3* __restrict__ forces,
        const int* __restrict__ starts,
        const int* __restrict__ cells
    );

} // namespace ppb::cuda::nbody