#pragma once

namespace ppb {
    __constant__ size_t numParticles;
    
    __constant__ size_t frequency;

    __constant__ float deltaT;

    __constant__ float cutoff_radius;

    __constant__ float verlet_skin;

    __constant__ float globalForce[3];
} // namespace ppb