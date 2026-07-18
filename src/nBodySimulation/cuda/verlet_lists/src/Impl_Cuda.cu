#include "Impl_Cuda.cuh"
#include "kernels.cuh"
#include "common.cuh"
#include "constants.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define CHECK_CUDA_ERROR(val) ppb::check((val), #val, __FILE__, __LINE__)

namespace ppb {
    template<typename FloatType>
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) 
        : _config{config}
    {
        //-------------------------------------Init constant memory----------------------------------------
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(numParticles, &_config.size, sizeof(_config.size)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(deltaT, &_config.deltaT, sizeof(_config.deltaT)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cutoff_radius, &_config.cutoff_radius, sizeof(_config.cutoff_radius)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(verlet_skin, &_config.verlet_skin, sizeof(_config.verlet_skin)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(globalForce, _config.globalForce.data(), sizeof(_config.globalForce)));
#ifdef PPB_ENABLE_VERLET_CLUSTER_LISTS
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(boxMin, _config.boxMin.data(), sizeof(_config.boxMin)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(boxMax, _config.boxMax.data(), sizeof(_config.boxMax)));
#endif 
        //------------------------------Determine optimal grid size----------------------------------------
        size = _config.size;
        frequency = _config.frequency;
        constexpr unsigned int WARP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;
        if (size <= MAX_THREADS) {
            _blockSize = size;
        } else {
            int minGridSize = 0;
            CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, reinterpret_cast<void *>(update_positions), 0, size));
        }
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize);

        //---------------------------------Allocate device memory------------------------------------------
        CHECK_CUDA_ERROR(cudaMalloc(&starts, sizeof(int) * (size + 1))); // +1 so the last element is the total number of neighbors
        CHECK_CUDA_ERROR(cudaMemset(starts, 0, sizeof(int) * (size + 1)));
    }

    template<typename FloatType>
    ImplCuda<FloatType>::~ImplCuda() {
        CHECK_CUDA_ERROR(cudaFree(starts));
        CHECK_CUDA_ERROR(cudaFree(verletLists));
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updatePositionsAndResetForce() {
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start));
        update_positions<<<_gridSize, _blockSize>>>(position, velocity, force, oldForce); 
        // every frequency iterations update verlet lists
        if (iteration % frequency == 0) {
#ifdef PPB_ENABLE_VERLET_CLUSTER_LISTS 
            // TODO max occupancy for those kernel launches with magic numbers
            // 1) Make clusters of size M and N by spatial gridding and binning
            float volume_domain = (boxMax[0] - boxMin[0]) * (boxMax[1] - boxMin[1]) * (boxMax[2] - boxMin[2]);
            float rho = volume_domain / numParticles;
            float tower_size = std::pow(max(M, N) / rho, 1.0/3.0);
            count_elements_in_towers<<<_gridSize, _blockSize>>>(position, starts_towers, tower_size); 
            add_dummy_particles_to_towers<<<util::ceilDiv(sizeof(starts_towers - 1), 1024), 1024>>>(starts_towers, M); //might not be the optimal grid size but i dont think this really matters here.
            thrust::inclusive_scan(thrust::device, starts_towers, starts_towers + sizeof(starts_towers), starts_towers);
            size_t size_clustersM = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&size_clustersM, &starts[size], sizeof(int), cudaMemcpyDeviceToHost)); 
            if (size_clustersM == 0) return;
            CHECK_CUDA_ERROR(cudaMalloc(&clustersM, sizeof(int) * size_clustersM));
            insert_particles_into_towers<<<_gridSize, _blockSize>>>(position, clustersM, starts_towers, tower_size);
            sort_particles_along_z_axis<<<util::ceilDiv(sizeof(starts_towers - 1), 1024), 1024>>>();
            
            // 2) Compute the bounding boxes
            compute_bounding_boxes<<<util::ceilDiv(sizeof(clustersM) / M, 1024), 1024>>>(BBM, clustersM, M);
            compute_bounding_boxes<<<util::ceilDiv(sizeof(clustersM) / N, 1024), 1024>>>(BBN, clustersM, N);
            
            // 3) Create the first preliminary cluster pair list
            cluster_pair_search<<<util::ceilDiv(sizeof(clustersM) / M, 1024), 1024>>>(BBM, BBN, cluster_pairs, starts);
           
            // 4) Prune the cluster pair list
            size_t gridSizePrune;
            size_t blockSizePrune;
            cluster_pair_prune<<<gridSizeSearch, blockSizeSearch>>>(clusters, pair_lists, starts); 
#else
            if (verletLists != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(verletLists));
            }
            get_number_of_neighbors<<<_gridSize, _blockSize>>>(starts, position);
            thrust::inclusive_scan(thrust::device, starts, starts + (size + 1), starts);
            size_t num_neighbors = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&num_neighbors, &starts[size], sizeof(int), cudaMemcpyDeviceToHost));
            if (num_neighbors != 0) {
                CHECK_CUDA_ERROR(cudaMalloc(&verletLists, sizeof(int) * num_neighbors));
                make_verlet_lists<<<_gridSize, _blockSize>>>(verletLists, starts, position);
            }
        }
#endif
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
        compute_forces<<<_gridSize, _blockSize>>>(position, force, verletLists, starts);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        _timings.forceUpdateTime += (elapsedTime * 16);
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

        for (iteration = 0; iteration < _config.numberTimeSteps; ++iteration) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }


    template class ImplCuda<float>;
  } // namespace ppb
