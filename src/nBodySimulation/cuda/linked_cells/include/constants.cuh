#pragma once

namespace ppb::cuda::nbody {
    __constant__ size_t NUM_PARTICLES;

    __constant__ int X_DIM;

    __constant__ int Y_DIM;

    __constant__ int Z_DIM;

    __constant__ int OFFSETS[27];

    __constant__ int OFFSETS_COLORED[8]; 
    
    __constant__ int OFFSETS_COLORED_NON_BASE_CELL[12]; 

    __constant__ int NUM_CELLS_SAME_COLOR;
    
    __constant__ int X_DIM_NEAREST_4;
    
    __constant__ int Y_DIM_NEAREST_4;
    
    __constant__ int Z_DIM_NEAREST_4;

    __constant__ float DELTA_T; 
    
    __constant__ float CUTOFF_RADIUS;

    __constant__ float CELL_SIZE;

    __constant__ float BOX_MIN[3];

    __constant__ float GLOBAL_FORCE[3];
} // namespace ppb::cuda::nbody