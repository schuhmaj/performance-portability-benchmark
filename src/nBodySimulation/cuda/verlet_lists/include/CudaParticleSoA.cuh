#pragma once

#include "nBodySimulation/Particle.h"

namespace ppb {
    template <typename FloatType>
    struct CudaParticleSoA {
        const std::vector<Particle<FloatType>> &_ref;

        //Device memory
        float3* positions{nullptr};
        float3* velocities{nullptr};
        float3* forces{nullptr};
        float3* oldForces{nullptr};
        
        //Host memory
        std::vector<float3> positionsHost;
        std::vector<float3> velocitiesHost;
        std::vector<float3> forcesHost;

        explicit CudaParticleSoA(const std::vector<Particle<FloatType>> &particles);

        ~CudaParticleSoA();

        std::vector<Particle<FloatType>> toParticles();
    };
} // namespace ppb