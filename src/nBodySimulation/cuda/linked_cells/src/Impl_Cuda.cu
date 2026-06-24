#include "Impl_Cuda.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

//taken from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/ (last accessed 13.6.26, 19:44)
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

namespace ppb {
    template<typename FloatType>
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) 
        : _config{config}
    {
        //DO NOT PUT THESE LINES IN THE CONSTRUCTOR CAUSE IT BREAKS FOR SOME WEIRD REASON
        _globalForce.x = _config.globalForce[0];
        _globalForce.y = _config.globalForce[1];
        _globalForce.z = _config.globalForce[2];
        //-------------------------------------------------------------------------------
 
        const size_t size = _config.size;
        constexpr unsigned int WARP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;

        int minGridSize = 0;
        CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, reinterpret_cast<void *>(update_positions<FloatType>), 0, size));
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize); 

        x_dim = (_config.boxMax[0] - _config.boxMin[0]) / _config.cell_size;
        y_dim = (_config.boxMax[1] - _config.boxMin[1]) / _config.cell_size;
        z_dim = (_config.boxMax[2] - _config.boxMin[2]) / _config.cell_size;
        

#ifdef PPB_ENABLE_DOMAIN_COLORING
        size_t number_of_cells_with_same_color = util::ceilDiv<size_t>(x_dim * y_dim * z_dim, 8);
        int offsets_coloredDeclared[8] = { 13, 14, 16, 17, 22, 23, 25, 26 };
        memcpy(offsets_colored, &offsets_coloredDeclared, 8 * sizeof(int));
        if (number_of_cells_with_same_color <= MAX_THREADS) {
            _blockSizeColored = number_of_cells_with_same_color;
        } else {
            CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&_gridSizeColored, &_blockSizeColored, reinterpret_cast<void *>(compute_forces_colored), 0, 0));
        }
        _gridSizeColored = util::ceilDiv<unsigned int>(number_of_cells_with_same_color, _blockSizeColored);
        std::cout<<"_gridSizeColored: "<<_gridSizeColored<<", _blockSizeColored: "<<_blockSizeColored<<std::endl;
#endif
        int offsetsDeclared[27] = {
            //front section
            -((x_dim + 1) * y_dim) - 1, -((x_dim + 1) * y_dim), -((x_dim + 1) * y_dim) + 1,
            -(x_dim * y_dim) - 1, -(x_dim * y_dim), (x_dim * y_dim) + 1,
            -((x_dim - 1) * y_dim) - 1, -((x_dim - 1) * y_dim), -((x_dim - 1) * y_dim) + 1,
            //mid section
            -x_dim - 1, -x_dim, -x_dim + 1,
            -1, 0, 1,
            x_dim - 1, x_dim, x_dim + 1,
            //back section
            ((x_dim - 1) * y_dim) - 1, ((x_dim - 1) * y_dim), ((x_dim - 1) * y_dim) + 1,
            (x_dim * y_dim) - 1, (x_dim * y_dim), (x_dim * y_dim) + 1,
            ((x_dim + 1) * y_dim) - 1, ((x_dim + 1) * y_dim), ((x_dim + 1) * y_dim) + 1
        };
        memcpy(offsets, &offsetsDeclared, 27 * sizeof(int));
        
        const size_t num_cells = x_dim * y_dim * z_dim;        
        CHECK_CUDA_ERROR(cudaMalloc(&cells, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&tmp, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&cell_offsets, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&starts, sizeof(int) * (num_cells + 1)));
        CHECK_CUDA_ERROR(cudaMemset(cells, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(tmp, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(cell_offsets, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(starts, 0, sizeof(int) * (num_cells + 1)));
    }

    template<typename FloatType>
    ImplCuda<FloatType>::~ImplCuda() {
        CHECK_CUDA_ERROR(cudaFree(starts));
        CHECK_CUDA_ERROR(cudaFree(cells));
        CHECK_CUDA_ERROR(cudaFree(cell_offsets));
        CHECK_CUDA_ERROR(cudaFree(tmp));
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        const float cell_size = _config.cell_size;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        const auto &globalForce = _config.globalForce;
        const FloatType boxMinX = _config.boxMin[0];
        const FloatType boxMinY = _config.boxMin[1];
        const FloatType boxMinZ = _config.boxMin[2];
        const FloatType boxMaxX = _config.boxMax[0];
        const FloatType boxMaxY = _config.boxMax[1];
        const FloatType boxMaxZ = _config.boxMax[2];
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;
        const size_t num_cells = x_dim * y_dim * z_dim;
        
        CHECK_CUDA_ERROR(cudaMemset(starts, 0.0, sizeof(int) * (num_cells + 1)));

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        update_positions<FloatType><<<_gridSize, _blockSize>>>(position, velocity, force, oldForce, tmp, cell_offsets, 
            starts, _globalForce, dt, size, cell_size, x_dim, y_dim, z_dim, boxMinX, boxMinY, boxMinZ, boxMaxX, boxMaxY, boxMaxZ);        
        thrust::inclusive_scan(thrust::device, starts, starts + (num_cells + 1), starts);  
        update_cells<<<_gridSize, _blockSize>>>(cells, tmp, cell_offsets, starts, size);  
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        _timings.positionUpdateForceResetTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::computeForces() {
        const size_t size = _config.size;
        const float cutoff_radius = _config.cutoff_radius;
        const float cell_size = _config.cell_size;
        const FloatType boxMinX = _config.boxMin[0];
        const FloatType boxMinY = _config.boxMin[1];
        const FloatType boxMinZ = _config.boxMin[2];
        const FloatType boxMaxX = _config.boxMax[0];
        const FloatType boxMaxY = _config.boxMax[1];
        const FloatType boxMaxZ = _config.boxMax[2];
        auto &force = _particles->forces;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        CHECK_CUDA_ERROR(cudaEventRecord(start));
#ifdef PPB_ENABLE_DOMAIN_COLORING
        for (size_t color = 0; color < 8; color++) {
            compute_forces_colored<<<_gridSizeColored, _blockSizeColored>>>(color, position, force, 
                cells, starts, size, offsets, offsets_colored, cell_size, cutoff_radius, x_dim, y_dim, z_dim);
        }
#else
        compute_forces<FloatType><<<_gridSize, _blockSize>>>(position, force, cells, starts, size, offsets, 
            cell_size, cutoff_radius, x_dim, y_dim, z_dim, boxMinX, boxMinY, boxMinZ, boxMaxX, boxMaxY, boxMaxZ);
#endif
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        _timings.forceUpdateTime += (elapsedTime * 16);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        update_velocities<<<_gridSize, _blockSize>>>(velocity, force, oldForce, dt, size);
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
  } // namespace ppb
