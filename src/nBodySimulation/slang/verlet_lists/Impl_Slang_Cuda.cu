#include "Impl_Slang_Cuda.cuh"
#include <cuda_runtime.h>

#define CHECK(X)                                                       \
    do {                                                               \
        CUresult err = (X);                                            \
        if (err != CUDA_SUCCESS) {                                     \
            const char* msg;                                           \
            cuGetErrorString(err, &msg);                               \
            fprintf(stderr,                                            \
                "CUDA Driver error at %s:%d (%s): %s\n",               \
                __FILE__, __LINE__, #X, msg);                          \
        }                                                                                                                     \
    } while (0)

// cudaError_t err = cudaDeviceSynchronize();
// if (err != cudaSuccess) {
//     printf("CUDA runtime error before setupLinkedModule: %s\n",
//         cudaGetErrorString(err));
// }

namespace ppb {

    template <typename FloatType>
    CudaParticleSoA<FloatType>::CudaParticleSoA(const std::vector<Particle<FloatType>> &particles, const ParticleSimulationConfig<FloatType> &config)
        : positionsHost{particles.size()}
        , velocitiesHost{particles.size()}
        , forcesHost{particles.size()}
        , _ref{particles}
        , _config{config}
    {
        const size_t size = particles.size();
        for (size_t i = 0; i < size; ++i) {
            positionsHost[i] = {particles[i].getPosition()[0], particles[i].getPosition()[1], particles[i].getPosition()[2], 0.0};
            velocitiesHost[i] = {particles[i].getVelocity()[0], particles[i].getVelocity()[1], particles[i].getVelocity()[2], 0.0};
            forcesHost[i] = {particles[i].getForce()[0], particles[i].getForce()[1], particles[i].getForce()[2], 0.0};
        }

        uint32_t TILE_SIZE = _config.TILE_SIZE;
        uint32_t nBlocks = (particles.size() + TILE_SIZE) / TILE_SIZE;
        neighborsLength = nBlocks * TILE_SIZE;

        CHECK(cuInit(0));

        CUdevice device;
        CHECK(cuDeviceGet(&device, 0));
        CHECK(cuCtxCreate(&context, nullptr, 0, device));

        CHECK(cuMemAlloc(&positions, sizeof(float4) * size));
        CHECK(cuMemAlloc(&velocities, sizeof(float4) * size));
        CHECK(cuMemAlloc(&forces, sizeof(float4) * size));
        CHECK(cuMemAlloc(&oldForces, sizeof(float4) * size));

        CHECK(cuMemAlloc(&neighbors, sizeof(uint32_t) * neighborsLength));

        CHECK(cuMemcpyHtoD(positions, positionsHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(velocities, velocitiesHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(forces, forcesHost.data(), sizeof(float4) * size));
        CHECK(cuMemsetD8(oldForces, 0, sizeof(float4) * size));

        CHECK(cuMemsetD8(neighbors, 0, sizeof(uint32_t) * neighborsLength));

    }   

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        CHECK(cuMemFree(positions));
        CHECK(cuMemFree(velocities));
        CHECK(cuMemFree(forces));
        CHECK(cuMemFree(oldForces));

        CHECK(cuMemFree(neighbors));
        CHECK(cuCtxDestroy(context));
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> CudaParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        CHECK(cuMemcpyDtoH(positionsHost.data(), positions, sizeof(float4) * _ref.size()));
        CHECK(cuMemcpyDtoH(velocitiesHost.data(), velocities, sizeof(float4) * _ref.size()));
        CHECK(cuMemcpyDtoH(forcesHost.data(), forces, sizeof(float4) * _ref.size()));
        for (size_t i = 0; i < particles.size(); ++i) {
            const float4& position = positionsHost[i];
            const float4& velocity = velocitiesHost[i];
            const float4& force = forcesHost[i];
            particles[i].setPosition({position.x, position.y, position.z});
            particles[i].setVelocity({velocity.x, velocity.y, velocity.z});
            particles[i].setForce({force.x, force.y, force.z});
        }
        return particles;
    }

    template <typename FloatType>
    void CudaParticleSoA<FloatType>::print_buffer(CUdeviceptr buffer, size_t size) {
        CHECK(cuCtxSynchronize());
        std::vector<float4> host(size);
        CHECK(cuMemcpyDtoH(host.data(), buffer, sizeof(float4) * size));

        for (size_t i = 0; i < size; i++) {
            std::cout << i << ": "
                    << host[i].x << ", "
                    << host[i].y << ", "
                    << host[i].z << ", "
                    << host[i].w << "\n";
        }
    }

    template class CudaParticleSoA<float>;


    template<typename FloatType>
    ImplSlangCuda<FloatType>::ImplSlangCuda(const ParticleSimulationConfig<FloatType> &config) : _config{config}, _globalForce{_config.globalForce[0], _config.globalForce[1], _config.globalForce[2]} {
        _blockSize = _config.TILE_SIZE;
    }

    template <typename FloatType>
    void ImplSlangCuda<FloatType>::freeData(CUdeviceptr pc_ptr, CUmodule* module_) {
        CHECK(cuCtxSynchronize());
        CHECK(cuMemFree(pc_ptr));
        CHECK(cuModuleUnload(*module_));
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::freeExclusiveScanCache(ExclusiveScanCache* cache) {
        if (!cache) {
            return;
        }
        if (cache->cache) {
            freeExclusiveScanCache(cache->cache);
        }
        if (cache->module_blellochScan) {
            CHECK(cuModuleUnload(cache->module_blellochScan));
        }
        if (cache->module_blockSum) {
            CHECK(cuModuleUnload(cache->module_blockSum));
        }
        if (cache->pc) {
            CHECK(cuMemFree(cache->pc));
        }
        if (cache->blockSum) {
            CHECK(cuMemFree(cache->blockSum));
        }
        delete cache;
    }

    template <typename FloatType>
    CUdeviceptr ImplSlangCuda<FloatType>::setupKernel(void* pushData, CUmodule* module_, CUfunction* kernel, const char* file, const char* name, const char* params, size_t pushSize) {
        CHECK(cuModuleLoad(module_, file));
        CHECK(cuModuleGetFunction(kernel, *module_, name));
        CUdeviceptr memory;
        size_t size;
        CHECK(cuModuleGetGlobal(
            &memory,
            &size,
            *module_,
            params
        ));
        CHECK(cuMemcpyHtoD(memory, pushData, pushSize));
        return memory;
    }

    template<typename FloatType>
    ExclusiveScanCache* ImplSlangCuda<FloatType>::setupExclusiveScanCache(CUdeviceptr data, uint32_t totalLength) {
        uint32_t nBlocks = (totalLength + _blockSize - 1) / _blockSize;
        uint32_t blockSumSize = ((nBlocks + _blockSize - 1) / _blockSize) * _blockSize;
        // manage memory for exclusiveScan
        CUdeviceptr blockSum;
        CHECK(cuMemAlloc(&blockSum, sizeof(uint32_t) * blockSumSize));
        // instantiate push constants
        ExclusivePushConstants excl_pc_host{};
        excl_pc_host.total_size = totalLength;
        excl_pc_host.block_size = _blockSize;
        CUdeviceptr excl_pc_ptr;
        CHECK(cuMemAlloc(&excl_pc_ptr, sizeof(ExclusivePushConstants)));
        CHECK(cuMemcpyHtoD(excl_pc_ptr, &excl_pc_host, sizeof(ExclusivePushConstants)));
        // instantiate push data
        PushExclusive params_exclusiveScan{};
        params_exclusiveScan.data      = ResourceSlot{data, 0};
        params_exclusiveScan.blockSums = ResourceSlot{blockSum, 0};
        params_exclusiveScan.pc        = excl_pc_ptr;
        // setup blellochScan module and kernel
        CUmodule module_blellochScan;
        CUfunction kernel_blellochScan;
        setupKernel(&params_exclusiveScan, &module_blellochScan, &kernel_blellochScan, SLANG_PTX_DIR "/KernelBlellochScan.ptx", "computeBlellochScan", "Params_ExclusiveScan", sizeof(PushExclusive));

        ExclusiveScanCache* cache = new ExclusiveScanCache{};
        cache->pc                  = excl_pc_ptr;
        cache->blockSum            = blockSum;
        cache->module_blellochScan = module_blellochScan;
        cache->kernel_blellochScan = kernel_blellochScan;

        if (nBlocks > 1) {
            CUmodule module_blockSum;
            CUfunction kernel_blockSum;
            setupKernel(&params_exclusiveScan, &module_blockSum, &kernel_blockSum, SLANG_PTX_DIR "/KernelBlockSum.ptx", "computeBlockSum", "Params_ExclusiveScan", sizeof(PushExclusive));

            cache->module_blockSum = module_blockSum;
            cache->kernel_blockSum = kernel_blockSum;
            
            cache->cache = setupExclusiveScanCache(blockSum, blockSumSize);
        }
        else {
            cache->module_blockSum = nullptr;
            cache->kernel_blockSum = nullptr;
            cache->cache           = nullptr;
        }

        return cache;
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::exclusiveScanBlelloch(uint32_t totalLength, ExclusiveScanCache* cache) {
        uint32_t nBlocks = (totalLength + _blockSize - 1) / _blockSize;

        // dispatch blellochScan
        launchKernel(&cache->kernel_blellochScan, nBlocks, &_timings.neighborSearch);

        if (nBlocks > 1) {
            uint32_t blockSumSize = ((nBlocks + _blockSize - 1) / _blockSize) * _blockSize;
            // recursively calculate prefix sum
            exclusiveScanBlelloch(blockSumSize, cache->cache);
            // add block offset
            // dispatch blockSum
            launchKernel(&cache->kernel_blockSum, nBlocks, &_timings.neighborSearch);
        }
    }

    template<typename FloatType>
    CUdeviceptr ImplSlangCuda<FloatType>::createVerletList(CUdeviceptr verletList,
                                                    CUdeviceptr neighborsStarts, 
                                                    CUdeviceptr memory_verlet,
                                                    CUfunction* kernel_verlet,
                                                    PushVerlet* pushData_verlet,
                                                    CUdeviceptr memory_force,
                                                    PushFor* pushData_force,
                                                    const uint32_t gs)  {                                              
        CHECK(cuCtxSynchronize());
        CHECK(cuMemFree(verletList));  
        std::vector<uint32_t> data(_config.size);
        CHECK(cuMemcpyDtoH(data.data(), neighborsStarts, sizeof(uint32_t) * (_config.size + 1)));
        uint32_t nNeighbors = std::max(data[_config.size], 1u);

        if (nNeighbors > _config.size * _config.size) {
            std::cout << nNeighbors << "\n";
            uint32_t TILE_SIZE = _config.TILE_SIZE;
            uint32_t nBlocks = (_config.size + TILE_SIZE) / TILE_SIZE;
            print_buffer_uint(neighborsStarts, nBlocks * TILE_SIZE);
            exit(0);
        }

        CHECK(cuMemAlloc(&verletList, sizeof(uint32_t) * nNeighbors));
        pushData_verlet->verletLists = ResourceSlot{verletList, 0};
        CHECK(cuMemcpyHtoD(memory_verlet, pushData_verlet, sizeof(PushVerlet)));

        launchKernel(kernel_verlet, gs, &_timings.neighborSearch);

        pushData_force->verletLists = ResourceSlot{verletList, 0};
        CHECK(cuMemcpyHtoD(memory_force, pushData_force, sizeof(PushFor)));
        return verletList;
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::launchKernel(CUfunction* kernel, const uint32_t gs, double* timingField) {
        float elapsedTime;
        CUevent start, stop;
        CUstream stream;
        CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
        CHECK(cuStreamCreate(&stream, 0));

        CHECK(cuEventRecord(start, stream));
        CHECK(cuLaunchKernel(*kernel, gs, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
        CHECK(cuEventRecord(stop, stream));

        CHECK(cuEventSynchronize(stop));
        CHECK(cuEventElapsedTime(&elapsedTime, start, stop));
        *timingField += (elapsedTime * 1e6);
        
        CHECK(cuEventDestroy(start));
        CHECK(cuEventDestroy(stop));
        CHECK(cuStreamDestroy(stream));
    }

    template <typename FloatType>
    void ImplSlangCuda<FloatType>::print_buffer_uint(CUdeviceptr buffer, size_t size) {
        CHECK(cuCtxSynchronize());
        std::vector<uint32_t> host(size);
        CHECK(cuMemcpyDtoH(host.data(), buffer, sizeof(uint32_t) * size));

        for (size_t i = 0; i < size; i++) {
            std::cout << host[i] << ", ";
        }
        std::cout << "\n";
    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplSlangCuda<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles, _config);

        // =============================================================================
        //                               Kernel Parameters
        // =============================================================================

        // allocate a dummy verlet list for initial setup
        CUdeviceptr verletList;
        CHECK(cuMemAlloc(&verletList, sizeof(uint32_t) * 1));

        auto& soa = *_particles;

        // Parameters for KernelCountNeighbors.ptx
        CountPushConstants count_pc_host{};
        count_pc_host.n      = _config.size;
        count_pc_host.radius = _config.influenceRadius;
        // copying push constants
        CUdeviceptr count_pc_ptr;
        CHECK(cuMemAlloc(&count_pc_ptr, sizeof(CountPushConstants)));
        CHECK(cuMemcpyHtoD(count_pc_ptr, &count_pc_host, sizeof(CountPushConstants)));

        PushCount params_countNeighbors{};
        params_countNeighbors.positions  = ResourceSlot{soa.positions, 0};
        params_countNeighbors.nNeighbors = ResourceSlot{soa.neighbors, 0};
        params_countNeighbors.pc         = count_pc_ptr;

        // Parameters for KernelVerlet.ptx
        VerletPushConstants verlet_pc_host{};
        verlet_pc_host.total_size = _config.size;
        verlet_pc_host.radius     = _config.influenceRadius;
        // copying push constants
        CUdeviceptr verlet_pc_ptr;
        CHECK(cuMemAlloc(&verlet_pc_ptr, sizeof(VerletPushConstants)));
        CHECK(cuMemcpyHtoD(verlet_pc_ptr, &verlet_pc_host, sizeof(VerletPushConstants)));

        PushVerlet params_verlet{};
        params_verlet.positions       = ResourceSlot{soa.positions, 0};
        params_verlet.verletLists     = ResourceSlot{verletList, 0};
        params_verlet.neighborsStarts = ResourceSlot{soa.neighbors, 0};
        params_verlet.pc              = verlet_pc_ptr;

        // Parameters for KernelPosition.ptx
        PosPushConstants pos_pc_host{};
        pos_pc_host.globalForce_x = _config.globalForce[0];
        pos_pc_host.globalForce_y = _config.globalForce[1];
        pos_pc_host.globalForce_z = _config.globalForce[2];
        pos_pc_host.dt            = _config.deltaT;
        pos_pc_host.numParticles  = _config.size;
        // copying push constants
        CUdeviceptr pos_pc_ptr;
        CHECK(cuMemAlloc(&pos_pc_ptr, sizeof(PosPushConstants)));
        CHECK(cuMemcpyHtoD(pos_pc_ptr, &pos_pc_host, sizeof(PosPushConstants)));

        PushPos params_position{};
        params_position.positions  = ResourceSlot{soa.positions, 0};
        params_position.velocities = ResourceSlot{soa.velocities, 0};
        params_position.forces     = ResourceSlot{soa.forces, 0};
        params_position.oldForces  = ResourceSlot{soa.oldForces, 0};
        params_position.pc         = pos_pc_ptr;

        // Parameters for KernelVelocity.ptx
        VelPushConstants vel_pc_host{};
        vel_pc_host.dt = _config.deltaT;
        vel_pc_host.numParticles = _config.size;
        // copying push constants
        CUdeviceptr vel_pc_ptr;
        CHECK(cuMemAlloc(&vel_pc_ptr, sizeof(VelPushConstants)));
        CHECK(cuMemcpyHtoD(vel_pc_ptr, &vel_pc_host, sizeof(VelPushConstants)));

        PushVel params_velocity{};
        params_velocity.velocities = ResourceSlot{soa.velocities, 0};
        params_velocity.forces     = ResourceSlot{soa.forces, 0};
        params_velocity.oldForces  = ResourceSlot{soa.oldForces, 0};
        params_velocity.pc         = vel_pc_ptr;

        // Parameters for KernelForce.ptx
        ForPushConstants for_pc_host{};
        for_pc_host.numParticles = _config.size;
        // copying push constants
        CUdeviceptr for_pc_ptr;
        CHECK(cuMemAlloc(&for_pc_ptr, sizeof(ForPushConstants)));
        CHECK(cuMemcpyHtoD(for_pc_ptr, &for_pc_host, sizeof(ForPushConstants)));

        PushFor params_force{};
        params_force.positions       = ResourceSlot{soa.positions, 0};
        params_force.forces          = ResourceSlot{soa.forces, 0};
        params_force.verletLists     = ResourceSlot{verletList, 0};
        params_force.neighborsStarts = ResourceSlot{soa.neighbors, 0};
        params_force.pc              = for_pc_ptr;

        // =============================================================================

        CUmodule module_countNeighbors;
        CUfunction kernel_countNeighbors;
        setupKernel(&params_countNeighbors, &module_countNeighbors, &kernel_countNeighbors, SLANG_PTX_DIR "/KernelCountNeighbors.ptx", "computeCountNeighbors", "Params_CountNeighbors", sizeof(PushCount));

        CUmodule module_verlet;
        CUfunction kernel_verlet;
        CUdeviceptr memory_verlet = setupKernel(&params_verlet, &module_verlet, &kernel_verlet, SLANG_PTX_DIR "/KernelVerlet.ptx", "computeVerlet", "Params_Verlet", sizeof(PushVerlet));

        CUmodule module_position;
        CUfunction kernel_position;
        setupKernel(&params_position, &module_position, &kernel_position, SLANG_PTX_DIR "/KernelPosition.ptx", "computePosition", "Params_Position", sizeof(PushPos));

        CUmodule module_velocity;
        CUfunction kernel_velocity;
        setupKernel(&params_velocity, &module_velocity, &kernel_velocity, SLANG_PTX_DIR "/KernelVelocity.ptx", "computeVelocity", "Params_Velocity", sizeof(PushVel));

        CUmodule module_force;
        CUfunction kernel_force;
        CUdeviceptr memory_force = setupKernel(&params_force, &module_force, &kernel_force, SLANG_PTX_DIR "/KernelForce.ptx", "computeForce", "Params_Force", sizeof(PushFor));

        ExclusiveScanCache* excl_cache = setupExclusiveScanCache(soa.neighbors, soa.neighborsLength);

        const uint32_t _gridSize = util::ceilDiv<unsigned int>(_config.size, _blockSize);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            // here 10 is a magic number and should still be experimentally determined.
            if (i % 10 == 0) {
                launchKernel(&kernel_countNeighbors, _gridSize, &_timings.neighborSearch);
                exclusiveScanBlelloch(soa.neighborsLength, excl_cache);
                verletList = createVerletList(verletList, soa.neighbors, memory_verlet, &kernel_verlet, &params_verlet, memory_force, &params_force, _gridSize);
            }

            launchKernel(&kernel_position, _gridSize, &_timings.positionUpdateForceResetTime);
            launchKernel(&kernel_force, _gridSize, &_timings.forceUpdateTime);
            launchKernel(&kernel_velocity, _gridSize, &_timings.velocityUpdateTime);
        }
        //_particles->print_buffer(soa.positions, _config.size);

        CHECK(cuMemFree(verletList));
        freeData(count_pc_ptr, &module_countNeighbors);
        freeData(verlet_pc_ptr, &module_verlet);
        freeData(pos_pc_ptr, &module_position);
        freeData(vel_pc_ptr, &module_velocity);
        freeData(for_pc_ptr, &module_force);
        freeExclusiveScanCache(excl_cache);
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplSlangCuda<float>;

};
