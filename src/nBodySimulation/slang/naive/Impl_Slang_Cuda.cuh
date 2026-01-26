#pragma once

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include "cuda.h"

namespace ppb {


    template <typename FloatType>
    struct CudaParticleSoA {

        const std::vector<Particle<FloatType>> &_ref;

        float4 *positions{nullptr};
        float4 *velocities{nullptr};
        float4 *forces{nullptr};
        float4 *oldForces{nullptr};

        std::vector<float4> positionsHost;
        std::vector<float4> velocitiesHost;
        std::vector<float4> forcesHost;

        explicit CudaParticleSoA(const std::vector<Particle<FloatType>> &particles);

        ~CudaParticleSoA();

        std::vector<Particle<FloatType>> toParticles();

        void print_buffer(float4 *buffer, size_t size);
    };

    template <typename FloatType>
    class ImplSlangCuda {

        ParticleSimulationConfig<FloatType> _config;

        std::optional<CudaParticleSoA<FloatType>> _particles{std::nullopt};

        ParticleSimulationTimings _timings{};

        int _blockSize;
        int _gridSize;
        float3 _globalForce;

    public:
        using float_type = FloatType;


        explicit ImplSlangCuda(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Completes the setup of .ptx kernels
         *
         * @param pushData A pointer to a struct which contains the buffers and push constants for the kernel (defined in Kernel_Structs.cuh).
         * @param module_ The CUmodule object to be used.
         * @param kernel The CUfunction object to be used.
         * @param file The filepath to the .ptx file.
         * @param name A name used for this shader.
         * @param params The name of the parameters in the .ptx file.
         * @param pushSize The size of the pushData struct.
         */
        void setupKernel(void* pushData, CUmodule* module_, CUfunction* kernel, const char* file, const char* name, const char* params, size_t pushSize);

        /**
         * Runs the simulation for the configured total time using parallel Kokkos kernels to update
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
        void updatePositionsAndResetForce(CUfunction* kernel_position);

        /**
         * Updates velocities of all particles on the device based on forces before and after the integration
         * step, using parallel execution.
         */
        void updateVelocities(CUfunction* kernel_velocity);

        /**
         * Computes the inter-particle forces using the Lennard-Jones potential for all particles on the device,
         * accumulating the results in parallel.
         */
        void computeForces(CUfunction* kernel_force);
    };
} // namespace ppb
