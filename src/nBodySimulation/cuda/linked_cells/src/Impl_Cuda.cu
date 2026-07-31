#include "Impl_Cuda.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include "constants.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define CHECK_CUDA_ERROR(val) ppb::cuda::nbody::check((val), #val, __FILE__, __LINE__)

namespace ppb::cuda::nbody {   

    template<typename FloatType>
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) 
        : _config{config}
    {
        x_dim_h = util::ceilDiv((_config.boxMax[0] - _config.boxMin[0]), _config.cell_size);
        y_dim_h = util::ceilDiv((_config.boxMax[1] - _config.boxMin[1]), _config.cell_size);
        z_dim_h = util::ceilDiv((_config.boxMax[2] - _config.boxMin[2]), _config.cell_size);
        int offsets_h[27] = {
            //front section
            -((x_dim_h + 1) * y_dim_h) - 1, -((x_dim_h + 1) * y_dim_h), -((x_dim_h + 1) * y_dim_h) + 1,
            -(x_dim_h * y_dim_h) - 1, -(x_dim_h * y_dim_h), -(x_dim_h * y_dim_h) + 1,
            -((x_dim_h - 1) * y_dim_h) - 1, -((x_dim_h - 1) * y_dim_h), -((x_dim_h - 1) * y_dim_h) + 1,
            //mid section
            -x_dim_h - 1, -x_dim_h, -x_dim_h + 1,
            -1, 0, 1,
            x_dim_h - 1, x_dim_h, x_dim_h + 1,
            //back section
            ((x_dim_h - 1) * y_dim_h) - 1, ((x_dim_h - 1) * y_dim_h), ((x_dim_h - 1) * y_dim_h) + 1,
            (x_dim_h * y_dim_h) - 1, (x_dim_h * y_dim_h), (x_dim_h * y_dim_h) + 1,
            ((x_dim_h + 1) * y_dim_h) - 1, ((x_dim_h + 1) * y_dim_h), ((x_dim_h + 1) * y_dim_h) + 1
        };
        
        //-------------------------------------Init constant memory----------------------------------------
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(numParticles, &_config.size, sizeof(_config.size)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(x_dim, &x_dim_h, sizeof(x_dim_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(y_dim, &y_dim_h, sizeof(y_dim_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(z_dim, &z_dim_h, sizeof(z_dim_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(offsets, offsets_h, sizeof(offsets_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(deltaT, &_config.deltaT, sizeof(_config.deltaT)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cutoff_radius, &_config.cutoff_radius, sizeof(_config.cutoff_radius)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cell_size, &_config.cell_size, sizeof(_config.cell_size)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(boxMin, _config.boxMin.data(), sizeof(_config.boxMin)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(globalForce, _config.globalForce.data(), sizeof(_config.globalForce)));
        
        //------------------------------Determine optimal grid sizes----------------------------------------
        const size_t size = _config.size;
        constexpr unsigned int WARP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;
        int minGridSize = 0;
        CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, reinterpret_cast<void *>(update_positions<FloatType>), 0, 0));
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize); 

#ifdef PPB_ENABLE_DOMAIN_COLORING
        int offsets_colored_h[8] = { 13, 14, 16, 17, 22, 23, 25, 26 };
        int offsets_colored_non_base_cell_h[12] = { 14, 16, 14, 25, 14, 22, 16, 22, 16, 23, 17, 22 };
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(offsets_colored, offsets_colored_h, sizeof(offsets_colored_h))); 
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(offsets_colored_non_base_cell, offsets_colored_non_base_cell_h, sizeof(offsets_colored_non_base_cell_h))); 
        size_t number_of_cells_with_same_color = util::ceilDiv<size_t>(x_dim_h * y_dim_h * z_dim_h, 8);
        if (number_of_cells_with_same_color <= MAX_THREADS) {
            _blockSizeForces = number_of_cells_with_same_color;
        } else {
            CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSizeForces, reinterpret_cast<void *>(compute_forces_colored), 0, 0));
        }
        _gridSizeForces = util::ceilDiv<unsigned int>(number_of_cells_with_same_color, _blockSizeForces);
#elif PPB_ENABLE_CUDA_LINKED_CELL_OPTIMIZATION 
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(compute_forces_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSizeForces, reinterpret_cast<void *>(compute_forces_optimized), 24 * 1024, 0));
        _gridSizeForces = util::ceilDiv<unsigned int>(size, _blockSizeForces); 
#else 
        _gridSizeForces = _gridSize;
        _blockSizeForces = _blockSize;
#endif
        
        //---------------------------------Allocate device memory------------------------------------------
        const size_t num_cells = x_dim_h * y_dim_h * z_dim_h;        
        CHECK_CUDA_ERROR(cudaMalloc(&cells, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&tmp, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&cell_offsets, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&starts, sizeof(int) * (num_cells + 1)));
        CHECK_CUDA_ERROR(cudaMalloc(&cells_positions, sizeof(float4) * size));
        CHECK_CUDA_ERROR(cudaMemset(cells, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(tmp, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(cell_offsets, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(starts, 0, sizeof(int) * (num_cells + 1)));
        CHECK_CUDA_ERROR(cudaMemset(cells_positions, 0, sizeof(float4) * size));
    }

    template<typename FloatType>
    ImplCuda<FloatType>::~ImplCuda() {
        CHECK_CUDA_ERROR(cudaFree(starts));
        CHECK_CUDA_ERROR(cudaFree(cells));
        CHECK_CUDA_ERROR(cudaFree(cell_offsets));
        CHECK_CUDA_ERROR(cudaFree(tmp));
        CHECK_CUDA_ERROR(cudaFree(cells_positions));
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updatePositionsAndResetForce() {
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;
        const size_t num_cells = x_dim_h * y_dim_h * z_dim_h;
        
        CHECK_CUDA_ERROR(cudaMemset(starts, 0.0, sizeof(int) * (num_cells + 1)));
        
        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        update_positions<<<_gridSize, _blockSize>>>(position, velocity, force, oldForce, tmp, cell_offsets, starts);        
        thrust::inclusive_scan(thrust::device, starts, starts + (num_cells + 1), starts);  
        update_cells<<<_gridSize, _blockSize>>>(cells, tmp, cell_offsets, starts, cells_positions, position);  
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        _timings.positionUpdateForceResetTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::computeForces() {
        auto &force = _particles->forces;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        CHECK_CUDA_ERROR(cudaEventRecord(start));
#ifdef PPB_ENABLE_DOMAIN_COLORING
        for (size_t color = 0; color < 8; color++) 
            compute_forces_colored<<<_gridSizeForces, _blockSizeForces>>>(color, position, force, cells, starts); 
#elif PPB_ENABLE_CUDA_LINKED_CELL_OPTIMIZATION
        //CHECK_CUDA_ERROR(cudaFuncSetAttribute(compute_forces_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        size_t shmem_size = 24 * 1024;
        compute_forces_optimized<<<_gridSizeForces, _blockSizeForces, shmem_size>>>(cells_positions, force, starts, cells, shmem_size / sizeof(float3));
#else
        compute_forces<<<util::ceilDiv<size_t>(_config.size, (size_t)512), 512>>>(cells_positions, force, cells, starts);
#endif
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        _timings.forceUpdateTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updateVelocities() {
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        update_velocities<<<_gridSize, _blockSize>>>(velocity, force, oldForce);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

        _timings.velocityUpdateTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplCuda<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }


    template class ImplCuda<float>;
  } // namespace ppb::cuda::nbody
