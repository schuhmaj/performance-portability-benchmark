#pragma once

namespace ppb::cuda::nbody {
    __constant__ size_t NUM_PARTICLES;
    
    __constant__ size_t FREQUENCY;

    __constant__ float DELTA_T;

    __constant__ float CUTOFF_RADIUS;

    __constant__ float VERLET_SKIN;

    __constant__ float GLOBAL_FORCE[3];

    __constant__ float BOX_MIN[3];

    __constant__ float BOX_MAX[3];

    __constant__ int X_DIM;
    
    __constant__ int Y_DIM;
    
    __constant__ int Z_DIM;
 
    __constant__ int OFFSETS[27];

    __constant__ float CELL_SIZE;
} // namespace ppb::cuda::nbody