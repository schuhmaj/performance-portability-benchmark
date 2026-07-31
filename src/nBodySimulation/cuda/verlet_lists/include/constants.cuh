#pragma once

namespace ppb::cuda::nbody {
    __constant__ size_t numParticles;
    
    __constant__ size_t frequency;

    __constant__ float deltaT;

    __constant__ float cutoff_radius;

    __constant__ float verlet_skin;

    __constant__ float globalForce[3];

    __constant__ float boxMin[3];

    __constant__ float boxMax[3];
} // namespace ppb::cuda::nbody