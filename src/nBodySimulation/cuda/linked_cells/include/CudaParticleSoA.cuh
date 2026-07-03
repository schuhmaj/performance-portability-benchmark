#pragma once

#include "nBodySimulation/Particle.h"

namespace ppb {
    template <typename FloatType>
    struct CudaParticleSoA {
        const std::vector<Particle<FloatType>> &_ref;

        //Device memory
        float4* positions{nullptr};
        float4* velocities{nullptr};
        float4* forces{nullptr};
        float4* oldForces{nullptr};
        
        //Host memory
        std::vector<float4> positionsHost;
        std::vector<float4> velocitiesHost;
        std::vector<float4> forcesHost;

        explicit CudaParticleSoA(const std::vector<Particle<FloatType>> &particles);

        ~CudaParticleSoA();

        std::vector<Particle<FloatType>> toParticles();
    };
} // namespace ppb