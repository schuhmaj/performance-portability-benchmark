#pragma once

#include "common/cuda/Cuda_Float3_Arithmetic.cuh"
#include "constants.cuh"

namespace ppb::cuda::nbody {
//------------------------------------------------------------------HELPER FUNCTIONS---------------------------------------------------------------------
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

    //taken from: https://github.com/dangets/cuda_examples/blob/master/clamp_function.cu (last accessed 14.6.26)
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
    * @brief Updates 'cells' to match the new positions of the particles
    * @param cells The cells buffer that the sorted particles will be stored in
    * @param tmp A temporary buffer where the i-th element is the index of the cell that the i-th particle is contained inside of
    * @param cell_offsets Buffer storing the offsets of the particles within their respective cell.
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    * @param cell_positions A buffer that stores the positions of the sorted particles in 'cells'
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    */
    __global__ void update_cells(
        int* cells, 
        int* tmp, 
        int* cell_offsets, 
        int* starts,
        float3* cells_positions,
        float3* positions
    );

    /**
    * @brief Updates the positions of the particles
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    * @param velocities The velocities buffer. The i-th element is the velocity of the i-th particle
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param oldForces The buffer storing the previous iteration's forces. The i-th element is the force of the previous iterations of the i-th particle
    * @param tmp A temporary buffer where the i-th element is the index of the cell that the i-th particle is contained inside of
    * @param cell_offsets Buffer storing the offsets of the particles within their respective cell.
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    */
    __global__ void update_positions(
        float3* positions, 
        const float3* velocities, 
        float3* forces, 
        float3* oldForces, 
        int* tmp, 
        int* cell_offsets,
        int* starts 
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
    * @brief [UNOPTIMIZED] Updates the forces of the particles.
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param cells The cells buffer that the sorted particles are stored in 
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    */
    __global__ void compute_forces_unoptimized(
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    );

    /**
    * @brief [GM BROADCAST OPT] Updates the forces of the particles.
    * @param cell_positions A buffer that stores the positions of the sorted particles in 'cells'
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param cells The cells buffer that the sorted particles are stored in 
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    */
    __global__ void compute_forces(
        const float3* __restrict__ cells_positions,
        float3* __restrict__ forces,
        const int* __restrict__ cells,
        const int* __restrict__ starts 
    );

    /**
    * @brief [DOMAIN COLORING] Updates the forces of the particles.
    * @param color The current color
    * @param positions The positions buffer. The i-th element is the position of the i-th particle
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param cells The cells buffer that the sorted particles are stored in 
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    */
    __global__ void compute_forces_colored(
        int color,
        const float3* __restrict__ positions,
        float3* __restrict__ forces,
        int* cells,
        int* starts
    );

    /**
    * @brief Returns the index and offset of the next element in the neighborhood of a given base cell
    * @param current_idx is the current index inside of 'cells'
    * @param offset is the offset which we have to the base_idx
    * @param base_cell_idx is the index of the base cell
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    * @returns (x, y)-tuple where x = index of next element, y = offset of cell that contains next element
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

    /**
    * @brief Returns the number of particles contained in the 27-cell range of a given base cell
    * @param base_cell_idx is the index of the base cell
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    * @returns The number of particles contained in the 27-cell range of a given base cell
    */
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

    /**
    * @brief [SHARED MEMORY OPT] Updates the forces of the particles.
    * @param cell_positions A buffer that stores the positions of the sorted particles in 'cells'
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param cells The cells buffer that the sorted particles are stored in 
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    * @param shmem_size The amount of shared memory used in multiples of sizeof(float3)
    */
    __global__ void compute_forces_optimized(
        const float3* __restrict__ cells_positions,
        float3* __restrict__ forces,
        const int* __restrict__ starts,
        const int* __restrict__ cells,
        const int shmem_size //size of shared memory in float3
    );

    /**
    * @brief [SHARED MEMORY OPT ALT] Updates the forces of the particles.
    * @param cell_positions A buffer that stores the positions of the sorted particles in 'cells'
    * @param forces The forces buffer. The i-th element is the force of the i-th particle
    * @param cells The cells buffer that the sorted particles are stored in 
    * @param starts The starts buffer that indicates where in 'cells' the different cells start and end
    */
    __global__ void compute_forces_optimized_alt(
        const float3* __restrict__ cells_positions,
        float3* __restrict__ forces,
        const int* __restrict__ starts,
        const int* __restrict__ cells
    );

} // namespace ppb::cuda::nbody