#pragma once

namespace ppb::cuda::nbody {
    __constant__ inline size_t NUM_PARTICLES;

    __constant__ inline int X_DIM;

    __constant__ inline int Y_DIM;

    __constant__ inline int Z_DIM;

    __constant__ inline int OFFSETS[27];

    __constant__ inline int OFFSETS_XYZ[81];

    __constant__ inline int OFFSETS_COLORED[8]; 
    
    __constant__ inline int OFFSETS_COLORED_NON_BASE_CELL[12]; 

    __constant__ inline int NUM_CELLS_SAME_COLOR;
    
    __constant__ inline int X_DIM_NEAREST_4;
    
    __constant__ inline int Y_DIM_NEAREST_4;
    
    __constant__ inline int Z_DIM_NEAREST_4;

    __constant__ inline float DELTA_T; 
    
    __constant__ inline float CUTOFF_RADIUS_SQUARED;

    __constant__ inline float CELL_SIZE;

    __constant__ inline float BOX_MIN[3];

    __constant__ inline float GLOBAL_FORCE[3];
} // namespace ppb::cuda::nbody