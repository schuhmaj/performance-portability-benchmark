#pragma once

namespace ppb::cuda::nbody {
    __constant__ inline size_t NUM_PARTICLES;
    
    __constant__ inline float DELTA_T;
    
    __constant__ inline float CUTOFF_RADIUS;

    __constant__ inline float CUTOFF_RADIUS_SQUARED;

    __constant__ inline float VERLET_SKIN;

    __constant__ inline float GLOBAL_FORCE[3];

    __constant__ inline float BOX_MIN[3];

    __constant__ inline float BOX_MAX[3];

    __constant__ inline int X_DIM;
    
    __constant__ inline int Y_DIM;
    
    __constant__ inline int Z_DIM;
 
    __constant__ inline int OFFSETS[27];
    
    __constant__ inline int OFFSETS_XYZ[81];

    __constant__ inline float CELL_SIZE;
} // namespace ppb::cuda::nbody