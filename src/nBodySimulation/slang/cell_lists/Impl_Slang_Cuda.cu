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

        std::array<float, 3> boxMin = _config.boxMin;
        std::array<float, 3> boxMax = _config.boxMax;
        boxSize = { boxMax[0] - boxMin[0], boxMax[1] - boxMin[1], boxMax[2] - boxMin[2] };
        cellCounts = { 
            util::ceilDiv<int>(boxSize[0], _config.h), 
            util::ceilDiv<int>(boxSize[1], _config.h), 
            util::ceilDiv<int>(boxSize[2], _config.h) };
        uint32_t nCells = cellCounts[0] * cellCounts[1] * cellCounts[2] + 1;
        uint32_t TILE_SIZE = _config.TILE_SIZE;
        uint32_t nBlocks = (nCells + TILE_SIZE - 1) / TILE_SIZE;
        cellsLength = nBlocks * TILE_SIZE;

        CHECK(cuInit(0));

        CUdevice device;
        CHECK(cuDeviceGet(&device, 0));
        CHECK(cuCtxCreate(&context, nullptr, 0, device));

        CHECK(cuMemAlloc(&positions, sizeof(float4) * size));
        CHECK(cuMemAlloc(&velocities, sizeof(float4) * size));
        CHECK(cuMemAlloc(&forces, sizeof(float4) * size));
        CHECK(cuMemAlloc(&oldForces, sizeof(float4) * size));

        CHECK(cuMemAlloc(&cells, sizeof(uint32_t) * cellsLength));
        CHECK(cuMemAlloc(&particleIdx, sizeof(int2) * size));
        CHECK(cuMemAlloc(&idCells, sizeof(uint32_t) * size));

        CHECK(cuMemcpyHtoD(positions, positionsHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(velocities, velocitiesHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(forces, forcesHost.data(), sizeof(float4) * size));
        CHECK(cuMemsetD8(oldForces, 0, sizeof(float4) * size));

        CHECK(cuMemsetD8(cells, 0, sizeof(uint32_t) * cellsLength));
        CHECK(cuMemsetD8(particleIdx, 0, sizeof(int2) * size));
        CHECK(cuMemsetD8(idCells, 0, sizeof(uint32_t) * size));

    }   

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        CHECK(cuMemFree(positions)); 
        CHECK(cuMemFree(velocities));
        CHECK(cuMemFree(forces));
        CHECK(cuMemFree(oldForces));

        CHECK(cuMemFree(cells));
        CHECK(cuMemFree(particleIdx));
        CHECK(cuMemFree(idCells));
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
    void ImplSlangCuda<FloatType>::setupKernel(void* pushData, CUmodule* module_, CUfunction* kernel, const char* file, const char* name, const char* params, size_t pushSize) {
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

        auto& soa = *_particles;

        // Parameters for KernelResetCells.ptx
        ResetPushConstants reset_pc_host{};
        reset_pc_host.total_size = soa.cellsLength;
        // copying push constants
        CUdeviceptr reset_pc_ptr;
        CHECK(cuMemAlloc(&reset_pc_ptr, sizeof(ResetPushConstants)));
        CHECK(cuMemcpyHtoD(reset_pc_ptr, &reset_pc_host, sizeof(ResetPushConstants)));

        PushReset params_resetCells{};
        params_resetCells.cells = ResourceSlot{soa.cells, 0};
        params_resetCells.pc    = reset_pc_ptr;

        // Parameters for KernelHistogram.ptx
        HistPushConstants hist_pc_host{};
        hist_pc_host.numParticles = _config.size;
        hist_pc_host.cCount_x     = soa.cellCounts[0];
        hist_pc_host.cCount_y     = soa.cellCounts[1];
        hist_pc_host.cCount_z     = soa.cellCounts[2];
        hist_pc_host.bMin_x       = _config.boxMin[0];
        hist_pc_host.bMin_y       = _config.boxMin[1];
        hist_pc_host.bMin_z       = _config.boxMin[2];
        hist_pc_host.bSize_x      = soa.boxSize[0];
        hist_pc_host.bSize_y      = soa.boxSize[1];
        hist_pc_host.bSize_z      = soa.boxSize[2];
        // copying push constants
        CUdeviceptr hist_pc_ptr;
        CHECK(cuMemAlloc(&hist_pc_ptr, sizeof(HistPushConstants)));
        CHECK(cuMemcpyHtoD(hist_pc_ptr, &hist_pc_host, sizeof(HistPushConstants)));

        PushHist params_histogram{};
        params_histogram.positions     = ResourceSlot{soa.positions, 0};
        params_histogram.histogram     = ResourceSlot{soa.cells, 0};
        params_histogram.particleIdx   = ResourceSlot{soa.particleIdx, 0};
        params_histogram.pc            = hist_pc_ptr;

        // Parameters for KernelIdCells.ptx
        IdPushConstants id_pc_host{};
        id_pc_host.numParticles = _config.size;
        // copying push constants
        CUdeviceptr id_pc_ptr;
        CHECK(cuMemAlloc(&id_pc_ptr, sizeof(IdPushConstants)));
        CHECK(cuMemcpyHtoD(id_pc_ptr, &id_pc_host, sizeof(IdPushConstants)));

        PushId params_idCells{};
        params_idCells.particleIdx = ResourceSlot{soa.particleIdx, 0};
        params_idCells.idCells     = ResourceSlot{soa.idCells, 0};
        params_idCells.starts      = ResourceSlot{soa.cells, 0};
        params_idCells.pc          = id_pc_ptr;

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
        for_pc_host.cCount_x     = soa.cellCounts[0];
        for_pc_host.cCount_y     = soa.cellCounts[1];
        for_pc_host.cCount_z     = soa.cellCounts[2];
        // copying push constants
        CUdeviceptr for_pc_ptr;
        CHECK(cuMemAlloc(&for_pc_ptr, sizeof(ForPushConstants)));
        CHECK(cuMemcpyHtoD(for_pc_ptr, &for_pc_host, sizeof(ForPushConstants)));

        PushFor params_force{};
        params_force.positions   = ResourceSlot{soa.positions, 0};
        params_force.forces      = ResourceSlot{soa.forces, 0};
        params_force.particleIdx = ResourceSlot{soa.particleIdx, 0};
        params_force.starts      = ResourceSlot{soa.cells, 0};
        params_force.idCells     = ResourceSlot{soa.idCells, 0};
        params_force.pc          = for_pc_ptr;

        // =============================================================================

        CUmodule module_resetCells;
        CUfunction kernel_resetCells;
        setupKernel(&params_resetCells, &module_resetCells, &kernel_resetCells, SLANG_PTX_DIR "/KernelResetCells.ptx", "computeResetCells", "Params_ResetCells", sizeof(PushReset));

        CUmodule module_histogram;
        CUfunction kernel_histogram;
        setupKernel(&params_histogram, &module_histogram, &kernel_histogram, SLANG_PTX_DIR "/KernelHistogram.ptx", "computeHistogram", "Params_Histogram", sizeof(PushHist));

        CUmodule module_idCells;
        CUfunction kernel_idCells;
        setupKernel(&params_idCells, &module_idCells, &kernel_idCells, SLANG_PTX_DIR "/KernelIdCells.ptx", "computeIdCells", "Params_IdCells", sizeof(PushId));

        CUmodule module_position;
        CUfunction kernel_position;
        setupKernel(&params_position, &module_position, &kernel_position, SLANG_PTX_DIR "/KernelPosition.ptx", "computePosition", "Params_Position", sizeof(PushPos));

        CUmodule module_velocity;
        CUfunction kernel_velocity;
        setupKernel(&params_velocity, &module_velocity, &kernel_velocity, SLANG_PTX_DIR "/KernelVelocity.ptx", "computeVelocity", "Params_Velocity", sizeof(PushVel));

        CUmodule module_force;
        CUfunction kernel_force;
        setupKernel(&params_force, &module_force, &kernel_force, SLANG_PTX_DIR "/KernelForce.ptx", "computeForce", "Params_Force", sizeof(PushFor));

        ExclusiveScanCache* excl_cache = setupExclusiveScanCache(soa.cells, soa.cellsLength);

        const uint32_t _gridSizePerParticle = util::ceilDiv<unsigned int>(_config.size, _blockSize);
        const uint32_t _gridSizePerCell = util::ceilDiv<unsigned int>(soa.cellsLength, _blockSize);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            if (i % _config.interval_neighbor_search == 0) {
                launchKernel(&kernel_resetCells, _gridSizePerCell, &_timings.neighborSearch);
                launchKernel(&kernel_histogram, _gridSizePerParticle, &_timings.neighborSearch);
                exclusiveScanBlelloch(soa.cellsLength, excl_cache);
                launchKernel(&kernel_idCells, _gridSizePerParticle, &_timings.neighborSearch);
            }

            launchKernel(&kernel_position, _gridSizePerParticle, &_timings.positionUpdateForceResetTime);
            launchKernel(&kernel_force, _gridSizePerParticle, &_timings.forceUpdateTime);
            launchKernel(&kernel_velocity, _gridSizePerParticle, &_timings.velocityUpdateTime);
        }
        //_particles->print_buffer(soa.positions, _config.size);

        freeData(reset_pc_ptr, &module_resetCells);
        freeData(hist_pc_ptr, &module_histogram);
        freeData(id_pc_ptr, &module_idCells);
        freeData(pos_pc_ptr, &module_position);
        freeData(vel_pc_ptr, &module_velocity);
        freeData(for_pc_ptr, &module_force);
        freeExclusiveScanCache(excl_cache);
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplSlangCuda<float>;

};
