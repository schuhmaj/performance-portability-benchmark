#include "Impl_Cuda.cuh"
#include "kernels.cuh"
#include "common/cuda/Cuda_Error_Checking.cuh"
#include "constants.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

#define CHECK_CUDA_ERROR(val) ppb::cuda::nbody::check((val), #val, __FILE__, __LINE__)

namespace ppb::cuda::nbody {
    template<typename FloatType>
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) 
        : _config{config}
    {
        //-------------------------------------Init constant memory----------------------------------------
        float cutoff_radius_squared = _config.cutoff_radius * _config.cutoff_radius;
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(NUM_PARTICLES, &_config.size, sizeof(_config.size)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(DELTA_T, &_config.deltaT, sizeof(_config.deltaT)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(CUTOFF_RADIUS, &_config.cutoff_radius, sizeof(_config.cutoff_radius)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(CUTOFF_RADIUS_SQUARED, &cutoff_radius_squared, sizeof(cutoff_radius_squared)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(VERLET_SKIN, &_config.verlet_skin, sizeof(_config.verlet_skin)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(GLOBAL_FORCE, _config.globalForce.data(), sizeof(_config.globalForce)));
#if defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS || defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS_OPT
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(BOX_MIN, _config.boxMin.data(), sizeof(_config.boxMin)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(BOX_MAX, _config.boxMax.data(), sizeof(_config.boxMax)));
#elif PPB_ENABLE_CUDA_VERLET_LISTS_LC_OPTIMIZATION
        int x_dim_h = util::ceilDiv((_config.boxMax[0] - _config.boxMin[0]), _config.cell_size);
        int y_dim_h = util::ceilDiv((_config.boxMax[1] - _config.boxMin[1]), _config.cell_size);
        int z_dim_h = util::ceilDiv((_config.boxMax[2] - _config.boxMin[2]), _config.cell_size);

        int offsets_xyz[81] = {
            //front section
            -1, -1, -1,     0, -1, -1,      1, -1, -1,
            -1, 0, -1,      0, 0, -1,       1, 0, -1,
            -1, 1, -1,      0, 1, -1,       1, 1, -1,

            //mid section
            -1, -1, 0,      0, -1, 0,       1, -1, 0,
            -1, 0, 0,       0, 0, 0,        1, 0, 0,
            -1, 1, 0,       0, 1, 0,        1, 1, 0,
            
            //back section
            -1, -1, 1,      0, -1, 1,       1, -1, 1,
            -1, 0, 1,       0, 0, 1,        1, 0, 1,
            -1, 1, 1,       0, 1, 1,        1, 1, 1,
        };

        int offsets_h[27];
        for (int o = 0; o < 27; o++) {
            offsets_h[o] = offsets_xyz[3*o] 
                         + offsets_xyz[3*o + 1] * x_dim_h
                         + offsets_xyz[3*o + 2] * x_dim_h * y_dim_h;
        }

        float cell_size = _config.cutoff_radius + _config.verlet_skin;
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(X_DIM, &x_dim_h, sizeof(x_dim_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Y_DIM, &y_dim_h, sizeof(y_dim_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Z_DIM, &z_dim_h, sizeof(z_dim_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(OFFSETS, offsets_h, sizeof(offsets_h)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(OFFSETS_XYZ, offsets_xyz, sizeof(offsets_xyz)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(CELL_SIZE, &cell_size, sizeof(cell_size)));
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(BOX_MIN, _config.boxMin.data(), sizeof(_config.boxMin)));
#endif
        //------------------------------Determine optimal grid size----------------------------------------
        size = _config.size;
        frequency = _config.interval_neighbor_search;
        constexpr unsigned int WARP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;
        if (size <= MAX_THREADS) {
            _blockSize = size;
        } else {
            int minGridSize = 0;
            CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, reinterpret_cast<void *>(update_positions), 0, size));
        }
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize);

#if defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS || defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS_OPT
        int minGridSize = 0;
        CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSizeForces, reinterpret_cast<void *>(compute_force_cluster_lists), 0, 0));
#endif
        //---------------------------------Allocate device memory------------------------------------------
#if !defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS && !defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS_OPT
        CHECK_CUDA_ERROR(cudaMalloc(&starts, sizeof(int) * (size + 1))); // +1 so the last element is the total number of neighbors
        CHECK_CUDA_ERROR(cudaMemset(starts, 0, sizeof(int) * (size + 1)));
#endif
#ifdef PPB_ENABLE_CUDA_VERLET_LISTS_LC_OPTIMIZATION
        num_cells = x_dim_h * y_dim_h * z_dim_h;        
        CHECK_CUDA_ERROR(cudaMalloc(&cells, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&tmp, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&cell_offsets, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&starts_LC, sizeof(int) * (num_cells + 1)));
        CHECK_CUDA_ERROR(cudaMemset(cells, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(tmp, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(cell_offsets, 0, sizeof(int) * size));
        CHECK_CUDA_ERROR(cudaMemset(starts_LC, 0, sizeof(int) * (num_cells + 1)));
#endif
    }

    template<typename FloatType>
    ImplCuda<FloatType>::~ImplCuda() {
        if (starts != nullptr) CHECK_CUDA_ERROR(cudaFree(starts));
        if (starts_towers != nullptr) CHECK_CUDA_ERROR(cudaFree(starts_towers));
        if (clusters != nullptr) CHECK_CUDA_ERROR(cudaFree(clusters));
        if (z_coordinates != nullptr) CHECK_CUDA_ERROR(cudaFree(z_coordinates));
        if (BBM != nullptr) CHECK_CUDA_ERROR(cudaFree(BBM));
        if (BBN != nullptr) CHECK_CUDA_ERROR(cudaFree(BBN));
        if (cluster_pairs != nullptr) CHECK_CUDA_ERROR(cudaFree(cluster_pairs));
        if (starts_LC != nullptr) CHECK_CUDA_ERROR(cudaFree(starts_LC));
        if (verletLists != nullptr) CHECK_CUDA_ERROR(cudaFree(verletLists));
        if (cells != nullptr) CHECK_CUDA_ERROR(cudaFree(cells));
        if (cell_offsets != nullptr) CHECK_CUDA_ERROR(cudaFree(cell_offsets));
        if (tmp != nullptr) CHECK_CUDA_ERROR(cudaFree(tmp));
    }
    
    template<typename FloatType>
    void ImplCuda<FloatType>::makeClusters() {
        // Split up the domain into towers
        float volume_domain = (_config.boxMax[0] - _config.boxMin[0]) * (_config.boxMax[1] - _config.boxMin[1]) * (_config.boxMax[2] - _config.boxMin[2]);
        float rho = _config.size / volume_domain;
        tower_size = std::pow(std::max(M, N) / rho, 1.0/3.0);
        size_t num_towers_x = std::ceil((_config.boxMax[0] - _config.boxMin[0]) / tower_size);
        size_t num_towers_y = std::ceil((_config.boxMax[1] - _config.boxMin[1]) / tower_size);
        num_towers = num_towers_x * num_towers_y;
        if (num_towers <= 0 || std::isnan(num_towers) || std::isinf(num_towers)) {
            std::cerr
            <<"Invalid number of towers was encountered in the Verlet Cluster Lists approach."
            <<"Change simulation hyperparameters or try another algorithm." 
            <<"Aborting."
            <<std::endl;
            exit(-1);
        }

        // Get number of particles + dummy particles in towers
        if (starts_towers != nullptr) CHECK_CUDA_ERROR(cudaFree(starts_towers));
        CHECK_CUDA_ERROR(cudaMalloc(&starts_towers, sizeof(int) * (num_towers + 1)));
        CHECK_CUDA_ERROR(cudaMemset(starts_towers, 0, sizeof(int) * (num_towers + 1)));
        count_particles_in_towers<<<_gridSize, _blockSize>>>(_particles->positions, starts_towers, tower_size); 
        add_dummy_particles_to_towers<<<util::ceilDiv(num_towers, (size_t)1024), 1024>>>(starts_towers, M, num_towers);  
        thrust::inclusive_scan(thrust::device, starts_towers, starts_towers + num_towers + 1, starts_towers);
        
        // Allocate space for clusters
        CHECK_CUDA_ERROR(cudaMemcpy(&size_clusters, &starts_towers[num_towers], sizeof(int), cudaMemcpyDeviceToHost)); 
        if (size_clusters == 0) return;
        if (clusters != nullptr) CHECK_CUDA_ERROR(cudaFree(clusters));
        if (z_coordinates != nullptr) CHECK_CUDA_ERROR(cudaFree(z_coordinates));
        CHECK_CUDA_ERROR(cudaMalloc(&clusters, sizeof(int) * size_clusters));
        CHECK_CUDA_ERROR(cudaMalloc(&z_coordinates, sizeof(float) * size_clusters));
        init_clusters_and_z_coordinates<<<util::ceilDiv(size_clusters, (size_t)1024), 1024>>>(clusters, z_coordinates, size_clusters);

        // Insert particles into towers and sort them along z-axis
        int* positions_in_tower = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&positions_in_tower, sizeof(int) * size)); 
        get_particle_position_in_tower<<<_gridSize, _blockSize>>>(_particles->positions, clusters, starts_towers, positions_in_tower, tower_size); 
        insert_particles_into_towers<<<_gridSize, _blockSize>>>(_particles->positions, clusters, z_coordinates, starts_towers, positions_in_tower, tower_size); 
        CHECK_CUDA_ERROR(cudaFree(positions_in_tower));
        
        // Sort particles along z axis
        // Code taken from https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSegmentedSort.html#id16 (Last accessed: 31.7.26, 22:47)
        int num_items = size_clusters;
        int num_segments = num_towers;
        int* d_offsets = starts_towers;
        float* d_keys_in = z_coordinates;
        float* d_keys_out = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d_keys_out, sizeof(float) * size_clusters));
        int* d_values_in = clusters;
        int* d_values_out = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&d_values_out, sizeof(int) * size_clusters));

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            num_items, num_segments, d_offsets, d_offsets + 1);

        // Allocate temporary storage
        CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Run sorting operation
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            num_items, num_segments, d_offsets, d_offsets + 1);


        clusters = d_values_out;
        z_coordinates = d_keys_out;

        CHECK_CUDA_ERROR(cudaFree(d_keys_in));
        CHECK_CUDA_ERROR(cudaFree(d_values_in));
        CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::boundingBoxes() {
        // Allocate the bounding box arrays
        if (BBM != nullptr) CHECK_CUDA_ERROR(cudaFree(BBM));
        if (BBN != nullptr) CHECK_CUDA_ERROR(cudaFree(BBN));
        CHECK_CUDA_ERROR(cudaMalloc(&BBM, sizeof(BoundingBox) * (size_clusters / M)));
        CHECK_CUDA_ERROR(cudaMalloc(&BBN, sizeof(BoundingBox) * (size_clusters / N)));
        CHECK_CUDA_ERROR(cudaMemset(BBM, 0, sizeof(BoundingBox) * (size_clusters / M)));
        CHECK_CUDA_ERROR(cudaMemset(BBN, 0, sizeof(BoundingBox) * (size_clusters / N)));

        // Compute the bounding boxes
        compute_bounding_boxes<<<util::ceilDiv(size_clusters / M, (size_t)1024), 1024>>>(BBM, clusters, _particles->positions, M, size_clusters / M);
        compute_bounding_boxes<<<util::ceilDiv(size_clusters / N, (size_t)1024), 1024>>>(BBN, clusters, _particles->positions, N, size_clusters / N);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::createPairList() {
        // Allocate 'starts', which will demarkate the borders between two pair lists in 'cluster_pairs'
        if (starts != nullptr) CHECK_CUDA_ERROR(cudaFree(starts));
        CHECK_CUDA_ERROR(cudaMalloc(&starts, sizeof(int) * (size_clusters / M + 1)));
        CHECK_CUDA_ERROR(cudaMemset(starts, 0, sizeof(int) * (size_clusters / M + 1)));

/*         printStartsTowers<<<1,1>>>(starts_towers, num_towers);
        printClusters<<<1,1>>>(clusters, size_clusters);
        printBB<<<1,1>>>(BBM, size_clusters / M); */
        
        // Allocate 'cluster_pairs' by first getting its size
#ifndef PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS_OPT
        cluster_pair_search<<<util::ceilDiv(size_clusters / M, (size_t)1024), 1024>>>(BBM, BBN, true, nullptr, starts, clusters, _particles->positions, tower_size, size_clusters);
#else
        cluster_pair_search_optimized<<<util::ceilDiv(size_clusters / M, (size_t)1024), 1024>>>(BBM, BBN, true, nullptr, starts, starts_towers, clusters, _particles->positions, tower_size, size_clusters);
#endif
        thrust::inclusive_scan(thrust::device, starts, starts + (size_clusters / M + 1), starts);
        size_t size_cluster_pairs = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(&size_cluster_pairs, &starts[size_clusters / M], sizeof(int), cudaMemcpyDeviceToHost)); 
        if (cluster_pairs != nullptr) CHECK_CUDA_ERROR(cudaFree(cluster_pairs));
        CHECK_CUDA_ERROR(cudaMalloc(&cluster_pairs, sizeof(int) * size_cluster_pairs));
   
        // Do the pair search
#ifndef PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS_OPT
        cluster_pair_search<<<util::ceilDiv(size_clusters / M, (size_t)1024), 1024>>>(BBM, BBN, false, cluster_pairs, starts, clusters, _particles->positions, tower_size, size_clusters); 
#else        
        cluster_pair_search_optimized<<<util::ceilDiv(size_clusters / M, (size_t)1024), 1024>>>(BBM, BBN, false, cluster_pairs, starts, starts_towers, clusters, _particles->positions, tower_size, size_clusters);
#endif
    /*         printPairList<<<1,1>>>(starts, size_clusters / M, cluster_pairs, size_cluster_pairs); */
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
 #if defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS || PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS_OPT
            // TODO max occupancy for those kernel launches with magic numbers (not too important as these are not a bottleneck).
            // TODO think about what we *really* need to reallocate every iteration. This becomes a bottleneck for small frequencies.
            makeClusters();
            boundingBoxes();
            createPairList();
            // Potentially prune the constructed cluster pair list?
#elif PPB_ENABLE_CUDA_VERLET_LISTS_LC_OPTIMIZATION
            if (verletLists != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(verletLists));
            }

            CHECK_CUDA_ERROR(cudaMemset(starts_LC, 0, sizeof(int) * (num_cells + 1)));
            sort_particles_into_cells<<<_gridSize, _blockSize>>>(position, tmp, cell_offsets, starts_LC);
            thrust::inclusive_scan(thrust::device, starts_LC, starts_LC + (num_cells + 1), starts_LC);  
            update_cells<<<_gridSize, _blockSize>>>(cells, tmp, cell_offsets, starts_LC, position);
            get_number_of_neighbors_LC_OPT<<<_gridSize, _blockSize>>>(starts, position, starts_LC, cells);
            thrust::inclusive_scan(thrust::device, starts, starts + (size + 1), starts);
            size_t num_neighbors = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&num_neighbors, &starts[size], sizeof(int), cudaMemcpyDeviceToHost));
            if (num_neighbors != 0) {
                CHECK_CUDA_ERROR(cudaMalloc(&verletLists, sizeof(int) * num_neighbors));
                make_verlet_lists_LC_OPT<<<_gridSize, _blockSize>>>(verletLists, starts, position, starts_LC, cells);
            }
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
#endif
        }
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
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
#if defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS || defined PPB_ENABLE_CUDA_VERLET_CLUSTER_LISTS_OPT
        int _gridSizeForces = util::ceilDiv<int>(4 * size_clusters, _blockSizeForces);
        compute_force_cluster_lists<<<_gridSizeForces, _blockSizeForces>>>(position, force, clusters, cluster_pairs, starts, size_clusters);
#else
        compute_forces<<<_gridSize, _blockSize>>>(position, force, verletLists, starts);
#endif 
        CHECK_CUDA_ERROR(cudaEventRecord(stop));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
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
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        _timings.velocityUpdateTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplCuda<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);

        for (iteration = 0; iteration < _config.numberTimeSteps; ++iteration) {
/*             std::cout<<"-----------------------ITERATION "<<iteration<<"--------------------------"<<std::endl; */
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
/*             int j = 0;
            for (auto& p : _particles->toParticles()) {
                if (j == 0) {
                    std::cout<<p<<std::endl;
                }
                j++;
            } */
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplCuda<float>;
  } // namespace ppb::cuda::nbody
