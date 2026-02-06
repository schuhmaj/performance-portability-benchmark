#pragma once

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include "cuda.h"
#include "Kernel_Structs.cuh"

namespace ppb {


    template <typename FloatType>
    struct CudaParticleSoA {

        const std::vector<Particle<FloatType>> &_ref;
        const ParticleSimulationConfig<FloatType> &_config;

        std::array<int, 3> cellCounts;
        std::array<float, 3> boxSize;
        uint32_t cellsLength;

        CUcontext context;

        CUdeviceptr positions;
        CUdeviceptr velocities;
        CUdeviceptr forces;
        CUdeviceptr oldForces;

        CUdeviceptr cells;
        CUdeviceptr particleIdx;
        CUdeviceptr idCells;

        std::vector<float4> positionsHost;
        std::vector<float4> velocitiesHost;
        std::vector<float4> forcesHost;

        explicit CudaParticleSoA(const std::vector<Particle<FloatType>> &particles, const ParticleSimulationConfig<FloatType> &config);

        ~CudaParticleSoA();

        std::vector<Particle<FloatType>> toParticles();

        void print_buffer(CUdeviceptr buffer, size_t size);
    };

    template <typename FloatType>
    class ImplSlangCuda {

        ParticleSimulationConfig<FloatType> _config;

        std::optional<CudaParticleSoA<FloatType>> _particles{std::nullopt};

        ParticleSimulationTimings _timings{};

        float3 _globalForce;

        uint32_t _blockSize;

    public:
        using float_type = FloatType;


        explicit ImplSlangCuda(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Frees one push constant pointer and unloads one CUmodule.
         *
         * @param pc_ptr The pointer to the push constants to be freed.
         * @param module_ The pointer to the module that is to be unloaded.
         */
        void freeData(CUdeviceptr pc_ptr, CUmodule* module_);

        void freeExclusiveScanCache(ExclusiveScanCache* cache);

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

        ExclusiveScanCache* setupExclusiveScanCache(CUdeviceptr data, uint32_t totalLength);

        void exclusiveScanBlelloch(uint32_t totalLength, ExclusiveScanCache* cache);

        /**
         * launches a kernel and updates a timing field.
         *
         * @param kernel The pointer to the CUfunction to be launched.
         * @param gs the gridsize used by the kernel.
         * @param timingField the pointer to the timing field to be updated.
         */
        void launchKernel(CUfunction* kernel, const uint32_t gs, double* timingField);

        void print_buffer_uint(CUdeviceptr buffer, size_t size);
    };
} // namespace ppb
