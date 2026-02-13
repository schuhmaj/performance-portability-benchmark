#include "Impl_Slang_Cuda.cuh"
#include <cuda_runtime.h>

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
        CHECK(cuCtxSetCurrent(context));

        positions  = new DeviceMemory(sizeof(float4) * size);
        velocities = new DeviceMemory(sizeof(float4) * size);
        forces     = new DeviceMemory(sizeof(float4) * size);
        oldForces  = new DeviceMemory(sizeof(float4) * size);

        neighbors  = new DeviceMemory(sizeof(uint32_t) * neighborsLength);

        CHECK(cuMemcpyHtoD(positions->ptr, positionsHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(velocities->ptr, velocitiesHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(forces->ptr, forcesHost.data(), sizeof(float4) * size));
        CHECK(cuMemsetD8(oldForces->ptr, 0, sizeof(float4) * size));

        CHECK(cuMemsetD8(neighbors->ptr, 0, sizeof(uint32_t) * neighborsLength));

    }   

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        delete positions;
        delete velocities;
        delete forces;
        delete oldForces;

        delete neighbors;

        CHECK(cuCtxDestroy(context));
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> CudaParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        CHECK(cuMemcpyDtoH(positionsHost.data(), positions->ptr, sizeof(float4) * _ref.size()));
        CHECK(cuMemcpyDtoH(velocitiesHost.data(), velocities->ptr, sizeof(float4) * _ref.size()));
        CHECK(cuMemcpyDtoH(forcesHost.data(), forces->ptr, sizeof(float4) * _ref.size()));
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

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::freeExclusiveScanCache(ExclusiveScanCache* cache) {
        if (!cache) {
            return;
        }
        if (cache->child) {
            freeExclusiveScanCache(cache->child);
        }
        if (cache->module_blellochScan) {
            delete cache->module_blellochScan;
        }
        if (cache->module_blockSum) {
            delete cache->module_blockSum;
        }
        if (cache->pc) {
            delete cache->pc;
        }
        if (cache->blockSum) {
            delete cache->blockSum;
        }
        delete cache;
    }

    template <typename FloatType>
    CUdeviceptr ImplSlangCuda<FloatType>::setupKernel(void* pushData, CUmodule* module_, CUfunction* kernel, const char* file, size_t pushSize) {
        CHECK(cuModuleLoad(module_, file));
        CHECK(cuModuleGetFunction(kernel, *module_, "computeMain"));
        CUdeviceptr memory;
        size_t size;
        CHECK(cuModuleGetGlobal(
            &memory,
            &size,
            *module_,
            "SLANG_globalParams"
        ));
        CHECK(cuMemcpyHtoD(memory, pushData, pushSize));
        return memory;
    }

    template<typename FloatType>
    ExclusiveScanCache* ImplSlangCuda<FloatType>::setupExclusiveScanCache(CUdeviceptr data, uint32_t totalLength) {
        uint32_t nBlocks = (totalLength + _blockSize - 1) / _blockSize;
        uint32_t blockSumSize = ((nBlocks + _blockSize - 1) / _blockSize) * _blockSize;
        // manage memory for exclusiveScan
        DeviceMemory* blockSum = new DeviceMemory(sizeof(uint32_t) * blockSumSize);
        // instantiate push constants
        ExclusivePushConstants excl_pc_host{};
        excl_pc_host.total_size = totalLength;
        excl_pc_host.block_size = _blockSize;
        DeviceMemory* excl_pc = new DeviceMemory(sizeof(ExclusivePushConstants));
        CHECK(cuMemcpyHtoD(excl_pc->ptr, &excl_pc_host, sizeof(ExclusivePushConstants)));
        // instantiate push data
        PushExclusive params_exclusiveScan{};
        params_exclusiveScan.data      = ResourceSlot{data, 0};
        params_exclusiveScan.blockSums = ResourceSlot{blockSum->ptr, 0};
        params_exclusiveScan.pc        = excl_pc->ptr;
        // setup blellochScan module and kernel
        DeviceModule* module_blellochScan = new DeviceModule{};
        setupKernel(&params_exclusiveScan, &module_blellochScan->mod, &module_blellochScan->kernel, SLANG_PTX_DIR "/KernelBlellochScan.ptx", sizeof(PushExclusive));

        ExclusiveScanCache* cache  = new ExclusiveScanCache{};
        cache->pc                  = excl_pc;
        cache->blockSum            = blockSum;
        cache->module_blellochScan = module_blellochScan;

        if (nBlocks > 1) {
            DeviceModule* module_blockSum = new DeviceModule{};
            setupKernel(&params_exclusiveScan, &module_blockSum->mod, &module_blockSum->kernel, SLANG_PTX_DIR "/KernelBlockSum.ptx", sizeof(PushExclusive));

            cache->module_blockSum = module_blockSum;
            
            cache->child = setupExclusiveScanCache(blockSum->ptr, blockSumSize);
        }
        else {
            cache->module_blockSum = nullptr;
            cache->child           = nullptr;
        }

        return cache;
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::exclusiveScanBlelloch(uint32_t totalLength, ExclusiveScanCache* cache) {
        uint32_t nBlocks = (totalLength + _blockSize - 1) / _blockSize;

        // dispatch blellochScan
        launchKernel(&cache->module_blellochScan->kernel, nBlocks, &_timings.neighborSearch);

        if (nBlocks > 1) {
            uint32_t blockSumSize = ((nBlocks + _blockSize - 1) / _blockSize) * _blockSize;
            // recursively calculate prefix sum
            exclusiveScanBlelloch(blockSumSize, cache->child);
            // add block offset
            // dispatch blockSum
            launchKernel(&cache->module_blockSum->kernel, nBlocks, &_timings.neighborSearch);
        }
    }

    template<typename FloatType>
    DeviceMemory* ImplSlangCuda<FloatType>::createVerletList(DeviceMemory* verletList,
                                                    CUdeviceptr neighborsStarts, 
                                                    CUdeviceptr memory_verlet,
                                                    CUfunction* kernel_verlet,
                                                    PushVerlet* pushData_verlet,
                                                    CUdeviceptr memory_force,
                                                    PushFor* pushData_force,
                                                    const uint32_t gs)  {                                              
        CHECK(cuCtxSynchronize());
        delete verletList;
        std::vector<uint32_t> data(_config.size + 1);
        CHECK(cuMemcpyDtoH(data.data(), neighborsStarts, sizeof(uint32_t) * (_config.size + 1)));
        uint32_t nNeighbors = std::max(data[_config.size], 1u);

        DeviceMemory* newVerletList = new DeviceMemory(sizeof(uint32_t) * nNeighbors);
        pushData_verlet->verletLists = ResourceSlot{newVerletList->ptr, 0};
        CHECK(cuMemcpyHtoD(memory_verlet, pushData_verlet, sizeof(PushVerlet)));

        launchKernel(kernel_verlet, gs, &_timings.neighborSearch);

        pushData_force->verletLists = ResourceSlot{newVerletList->ptr, 0};
        CHECK(cuMemcpyHtoD(memory_force, pushData_force, sizeof(PushFor)));
        return newVerletList;
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
        CHECK(cuLaunchKernel(*kernel, gs, 1, 1, _blockSize, 1, 1, 0, stream, nullptr, nullptr));
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
        DeviceMemory* verletList = new DeviceMemory(sizeof(uint32_t) * 1);

        auto& soa = *_particles;

        // Parameters for KernelCountNeighbors.ptx
        CountPushConstants count_pc_host{};
        count_pc_host.n      = _config.size;
        count_pc_host.radius = _config.influenceRadius;
        // copying push constants
        DeviceMemory count_pc(sizeof(CountPushConstants));
        CHECK(cuMemcpyHtoD(count_pc.ptr, &count_pc_host, sizeof(CountPushConstants)));

        PushCount params_countNeighbors{};
        params_countNeighbors.positions  = ResourceSlot{soa.positions->ptr, 0};
        params_countNeighbors.nNeighbors = ResourceSlot{soa.neighbors->ptr, 0};
        params_countNeighbors.pc         = count_pc.ptr;

        // Parameters for KernelVerlet.ptx
        VerletPushConstants verlet_pc_host{};
        verlet_pc_host.total_size = _config.size;
        verlet_pc_host.radius     = _config.influenceRadius;
        // copying push constants
        DeviceMemory verlet_pc(sizeof(VerletPushConstants));
        CHECK(cuMemcpyHtoD(verlet_pc.ptr, &verlet_pc_host, sizeof(VerletPushConstants)));

        PushVerlet params_verlet{};
        params_verlet.positions       = ResourceSlot{soa.positions->ptr, 0};
        params_verlet.verletLists     = ResourceSlot{verletList->ptr, 0};
        params_verlet.neighborsStarts = ResourceSlot{soa.neighbors->ptr, 0};
        params_verlet.pc              = verlet_pc.ptr;

        // Parameters for KernelPosition.ptx
        PosPushConstants pos_pc_host{};
        pos_pc_host.globalForce_x = _config.globalForce[0];
        pos_pc_host.globalForce_y = _config.globalForce[1];
        pos_pc_host.globalForce_z = _config.globalForce[2];
        pos_pc_host.dt            = _config.deltaT;
        pos_pc_host.numParticles  = _config.size;
        // copying push constants
        DeviceMemory pos_pc(sizeof(PosPushConstants));
        CHECK(cuMemcpyHtoD(pos_pc.ptr, &pos_pc_host, sizeof(PosPushConstants)));

        PushPos params_position{};
        params_position.positions  = ResourceSlot{soa.positions->ptr, 0};
        params_position.velocities = ResourceSlot{soa.velocities->ptr, 0};
        params_position.forces     = ResourceSlot{soa.forces->ptr, 0};
        params_position.oldForces  = ResourceSlot{soa.oldForces->ptr, 0};
        params_position.pc         = pos_pc.ptr;

        // Parameters for KernelVelocity.ptx
        VelPushConstants vel_pc_host{};
        vel_pc_host.dt = _config.deltaT;
        vel_pc_host.numParticles = _config.size;
        // copying push constants
        DeviceMemory vel_pc(sizeof(VelPushConstants));
        CHECK(cuMemcpyHtoD(vel_pc.ptr, &vel_pc_host, sizeof(VelPushConstants)));

        PushVel params_velocity{};
        params_velocity.velocities = ResourceSlot{soa.velocities->ptr, 0};
        params_velocity.forces     = ResourceSlot{soa.forces->ptr, 0};
        params_velocity.oldForces  = ResourceSlot{soa.oldForces->ptr, 0};
        params_velocity.pc         = vel_pc.ptr;

        // Parameters for KernelForce.ptx
        ForPushConstants for_pc_host{};
        for_pc_host.numParticles = _config.size;
        // copying push constants
        DeviceMemory for_pc(sizeof(ForPushConstants));
        CHECK(cuMemcpyHtoD(for_pc.ptr, &for_pc_host, sizeof(ForPushConstants)));

        PushFor params_force{};
        params_force.positions       = ResourceSlot{soa.positions->ptr, 0};
        params_force.forces          = ResourceSlot{soa.forces->ptr, 0};
        params_force.verletLists     = ResourceSlot{verletList->ptr, 0};
        params_force.neighborsStarts = ResourceSlot{soa.neighbors->ptr, 0};
        params_force.pc              = for_pc.ptr;

        // =============================================================================

        DeviceModule module_countNeighbors;
        setupKernel(&params_countNeighbors, &module_countNeighbors.mod, &module_countNeighbors.kernel, SLANG_PTX_DIR "/KernelCountNeighbors.ptx", sizeof(PushCount));

        DeviceModule module_verlet;
        CUdeviceptr memory_verlet = setupKernel(&params_verlet, &module_verlet.mod, &module_verlet.kernel, SLANG_PTX_DIR "/KernelVerlet.ptx", sizeof(PushVerlet));

        DeviceModule module_position;
        setupKernel(&params_position, &module_position.mod, &module_position.kernel, SLANG_PTX_DIR "/KernelPosition.ptx", sizeof(PushPos));

        DeviceModule module_velocity;
        setupKernel(&params_velocity, &module_velocity.mod, &module_velocity.kernel, SLANG_PTX_DIR "/KernelVelocity.ptx", sizeof(PushVel));

        DeviceModule module_force;
        CUdeviceptr memory_force = setupKernel(&params_force, &module_force.mod, &module_force.kernel, SLANG_PTX_DIR "/KernelForce.ptx", sizeof(PushFor));

        ExclusiveScanCache* excl_cache = setupExclusiveScanCache(soa.neighbors->ptr, soa.neighborsLength);

        const uint32_t _gridSize = util::ceilDiv<unsigned int>(_config.size, _blockSize);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            // here 10 is a magic number and should still be experimentally determined.
            if (i % _config.interval_neighbor_search == 0) {
                launchKernel(&module_countNeighbors.kernel, _gridSize, &_timings.neighborSearch);
                exclusiveScanBlelloch(soa.neighborsLength, excl_cache);
                verletList = createVerletList(verletList, soa.neighbors->ptr, memory_verlet, &module_verlet.kernel, &params_verlet, memory_force, &params_force, _gridSize);
            }

            launchKernel(&module_position.kernel, _gridSize, &_timings.positionUpdateForceResetTime);
            launchKernel(&module_force.kernel, _gridSize, &_timings.forceUpdateTime);
            launchKernel(&module_velocity.kernel, _gridSize, &_timings.velocityUpdateTime);
        }
        //_particles->print_buffer(soa.positions->ptr, _config.size);

        delete verletList;
        freeExclusiveScanCache(excl_cache);
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplSlangCuda<float>;

};
