#pragma once

#include <hip/hip_runtime.h>

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

#if FLOAT_BITS == 32
using VectorType3 = float3;
using VectorType4 = float4;
using float_type = float;
#elif FLOAT_BITS == 64
using VectorType3 = double3;
using VectorType4 = double4;
using float_type = double;
#endif

namespace ppb {


    template <typename FloatType>
    struct HipParticleSoA {

        const std::vector<Particle<FloatType>> &_ref;

        VectorType3 *positions{nullptr};
        VectorType3 *velocities{nullptr};
        VectorType3 *forces{nullptr};
        VectorType3 *oldForces{nullptr};

        std::vector<VectorType3> positionsHost;
        std::vector<VectorType3> velocitiesHost;
        std::vector<VectorType3> forcesHost;

        explicit HipParticleSoA(const std::vector<Particle<FloatType>> &particles);

        ~HipParticleSoA();

        std::vector<Particle<FloatType>> toParticles();
    };

    template <typename FloatType>
    class ImplHip {

        ParticleSimulationConfig<FloatType> _config;

        std::optional<HipParticleSoA<FloatType>> _particles{std::nullopt};

        ParticleSimulationTimings _timings{};

        int _blockSize;
        int _gridSize;
        VectorType3 _globalForce;

    public:
        using float_type = FloatType;


        explicit ImplHip(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Runs the simulation for the configured total time using parallel HIP kernels to update
         * positions, velocities, and compute forces at each step.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::vector<Particle<FloatType>> Final state of all particles after the simulation.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

        /**
         * Updates positions of all particles on the device using the velocity Verlet integrator,
         * and resets each particle's force to the configured global force in parallel.
         */
        void updatePositionsAndResetForce();

        /**
         * Updates velocities of all particles on the device based on forces before and after the integration
         * step, using parallel execution.
         */
        void updateVelocities();

        /**
         * Computes the inter-particle forces using the Lennard-Jones potential for all particles on the device,
         * accumulating the results in parallel.
         */
        void computeForces();
    };
} // namespace ppb
