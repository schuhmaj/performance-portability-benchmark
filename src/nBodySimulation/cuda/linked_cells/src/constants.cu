#include "constants.cuh"

namespace ppb {
    __constant__ size_t numParticles;

    __constant__ int x_dim;

    __constant__ int y_dim;

    __constant__ int z_dim;

    __constant__ int offsets[27];

    __constant__ int offsets_colored[8]; 

    __constant__ float deltaT; 

    __constant__ float cutoff_radius;

    __constant__ float cell_size;

    __constant__ float boxMin[3];

    __constant__ float globalForce[3];
} // namespace ppb