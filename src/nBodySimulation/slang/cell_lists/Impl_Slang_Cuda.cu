#include "Impl_Slang_Cuda.cuh"
#include "Kernel_Structs.cuh"
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
        }                                                              \
    } while (0)

namespace ppb {

    template <typename FloatType>
    CudaParticleSoA<FloatType>::CudaParticleSoA(const std::vector<Particle<FloatType>> &particles)
        : positionsHost{particles.size()}
        , velocitiesHost{particles.size()}
        , forcesHost{particles.size()}
        , _ref{particles}
    {
        const size_t size = particles.size();
        for (size_t i = 0; i < size; ++i) {
            positionsHost[i] = {particles[i].getPosition()[0], particles[i].getPosition()[1], particles[i].getPosition()[2], 0.0};
            velocitiesHost[i] = {particles[i].getVelocity()[0], particles[i].getVelocity()[1], particles[i].getVelocity()[2], 0.0};
            forcesHost[i] = {particles[i].getForce()[0], particles[i].getForce()[1], particles[i].getForce()[2], 0.0};
        }

        std::vector<uint32_t> cellsHost{nullptr};
        std::vector<int2> particleIdxHost{nullptr};
        std::vector<uint32_t> idCellsHost{nullptr};

        CHECK(cuInit(0));

        cudaMalloc(&positions, sizeof(float4) * size);
        cudaMalloc(&velocities, sizeof(float4) * size);
        cudaMalloc(&forces, sizeof(float4) * size);
        cudaMalloc(&oldForces, sizeof(float4) * size);

        std::array<float, 3> boxMin = _config.boxMin;
        std::array<float, 3> boxMax = _config.boxMax;
        std::array<float, 3> boxSize = { boxMax[0] - boxMin[0], boxMax[1] - boxMin[1], boxMax[2] - boxMin[2] };
        cellCounts = { 
            util::ceilDiv<int>(boxSize[0], _config.h), 
            util::ceilDiv<int>(boxSize[1], _config.h), 
            util::ceilDiv<int>(boxSize[2], _config.h) };
        uint32_t nCells = cellCounts[0] * cellCounts[1] * cellCounts[2] + 1;
        uint32_t TILE_SIZE = _config.TILE_SIZE;
        uint32_t nBlocks = (nCells + TILE_SIZE - 1) / TILE_SIZE;
        cellsLength = nBlocks * TILE_SIZE;

        cudaMalloc(&cells, sizeof(uint32_t) * cellsLength);
        cudaMalloc(&particleIdx, sizeof(int2) * size);
        cudaMalloc(&idCells, sizeof(uint32_t) * size);

        cudaMemcpy(positions, positionsHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(velocities, velocitiesHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(forces, forcesHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice);
        cudaMemset(oldForces, 0.0, sizeof(float4) * size);

        cudaMemset(cells, 0, sizeof(uint32_t) * cellsLength);
        cudaMemset(particleIdx, 0, sizeof(int2) * size);
        cudaMemset(idCells, 0, sizeof(uint32_t) * size);
    }   

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        cudaFree(positions);
        cudaFree(velocities);
        cudaFree(forces);
        cudaFree(oldForces);

        cudaFree(cells);
        cudaFree(particleIdx);
        cudaFree(idCells);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> CudaParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        cudaMemcpy(positionsHost.data(), positions, sizeof(float4) * _ref.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(velocitiesHost.data(), velocities, sizeof(float4) * _ref.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(forcesHost.data(), forces, sizeof(float4) * _ref.size(), cudaMemcpyDeviceToHost);
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
    void CudaParticleSoA<FloatType>::print_buffer(float4 *buffer, size_t size) {
        CHECK(cuCtxSynchronize());
        std::vector<float4> host(size);
        cudaMemcpy(host.data(), buffer, sizeof(float4) * size, cudaMemcpyDeviceToHost);

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
    ImplSlangCuda<FloatType>::freeData(CUdeviceptr pc_ptr, CUmodule* module_) {
        CHECK(cuCtxSynchronize());
        CHECK(cuMemFree(pc_ptr));
        CHECK(cuModuleUnload(*module_));
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

    template <typename FloatType>
    void ImplSlangCuda<FloatType>::setupLinkedModule(CUmodule* module_, CUfunction* kernel1, CUfunction* kernel2, const char* file1, const char* file2, const char* name1, const char* name2) {
        CUlinkState linkState;
        CHECK(cuLinkCreate(0, nullptr, nullptr, &linkState));
        CHECK(cuLinkAddFile(linkState, CU_JIT_INPUT_PTX, file1, 0, nullptr, nullptr));
        CHECK(cuLinkAddFile(linkState, CU_JIT_INPUT_PTX, file2, 0, nullptr, nullptr));

        void* cubin;
        size_t cubinSize;
        CHECK(cuLinkComplete(linkState, &cubin, &cubinSize));

        CHECK(cuModuleLoadData(&module_, cubin));
        CHECK(cuLinkDestroy(linkState));
        CHECK(cuModuleGetFunction(&kernel1, module_, name1));
        CHECK(cuModuleGetFunction(&kernel2, module_, name2));
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::computeResetCells(CUfunction* kernel_resetCells, const uint gs) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        CHECK(cuLaunchKernel(*kernel_resetCells, gs, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.neighborSearch += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::computeHistogram(CUfunction* kernel_histogram, const uint gs) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        CHECK(cuLaunchKernel(*kernel_histogram, gs, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.neighborSearch += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::exclusiveScanBlelloch(CUdeviceptr data, 
                                                         CUdeviceptr exclusiveScan_memory,
                                                         CUfunction* kernel_blellochScan, 
                                                         CUfunction* kernel_blockSum, 
                                                         uint totalLength) {
        uint32_t TILE_SIZE = _config.TILE_SIZE;
        uint32_t nBlocks = (totalLength + TILE_SIZE - 1) / TILE_SIZE;
        uint32_t blockSumSize = ((nBlocks + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

        // manage exclusiveScan memory
        ExclusivePushConstants excl_pc_host{};
        excl_pc_host.total_size = totalLength;
        excl_pc_host.block_size = blockSumSize;
        // copying push constants
        CUdeviceptr excl_pc_ptr;
        CHECK(cuMemAlloc(&excl_pc_ptr, sizeof(ExclusivePushConstants)));
        CHECK(cuMemcpyHtoD(excl_pc_ptr, &excl_pc_host, sizeof(ExclusivePushConstants)));

        std::vector<uint32_t> h(blockSumSize, 0);
        CUdeviceptr blockSum;
        cudaMalloc(&blockSum, sizeof(uint32_t) * blockSumSize);
        cudaMemcpy(blockSum, h.data(), sizeof(uint32_t) * blockSumSize, cudaMemcpyHostToDevice);
        PushBlelloch params_blellochScan{};
        params_blellochScan.data  = ResourceSlot{data, 0};
        params_blellochScan.blockSum = ResourceSlot{blockSum, 0};
        params_blellochScan.pc = excl_pc_ptr;
        CHECK(cuMemcpyHtoD(exclusiveScan_memory, params_blellochScan, sizeof(PushBlelloch)));

        // launch blellochScan kernel
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        CHECK(cuLaunchKernel(*kernel_blellochScan, nBlocks, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.neighborSearch += (elapsedTime * 1e6);

        // calculate prefix sum
        if (nBlocks > 1) {
            // setup module / kernels for next recusive call
            CUmodule module_exclusiveScan;
            CUfunction kernel_blellochScan;
            CUfunction kernel_blockSum;
            setupLinkedModule(module_exclusiveScan, kernel_blellochScan, kernel_blockSum, SLANG_PTX_DIR "/KernelBlellochScan.ptx", SLANG_PTX_DIR "/KernelBlockSum.ptx", "computeBlellochScan", "computeBlockSum");
            CUdeviceptr memory;
            size_t size;
            CHECK(cuModuleGetGlobal(&memory, &size, *module_, "Params_ExclusiveScan"));

            // next recursive call
            exclusiveScanBlelloch(blockSum, 
                                  memory, 
                                  kernel_blellochScan, 
                                  kernel_blockSum,
                                  blockSumSize);

            // launch blockSum kernel
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            CHECK(cuLaunchKernel(*kernel_blockSum, nBlocks, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            _timings.neighborSearch += (elapsedTime * 1e6);

            CHECK(cuCtxSynchronize());
            CHECK(cuModuleUnload(*module_exclusiveScan));
        }
        else {
            CHECK(cuCtxSynchronize());
        }
        CHECK(cuMemFree(excl_pc_ptr));
        cudaFree(blockSum);
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::computeIdCells(CUfunction* kernel_idCells, const uint gs) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        CHECK(cuLaunchKernel(*kernel_idCells, gs, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.neighborSearch += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::updatePositionsAndResetForce(CUfunction* kernel_position, const uint gs) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        CHECK(cuLaunchKernel(*kernel_position, gs, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.positionUpdateForceResetTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::updateVelocities(CUfunction* kernel_velocity, const uint gs) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        CHECK(cuLaunchKernel(*kernel_velocity, gs, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        _timings.velocityUpdateTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplSlangCuda<FloatType>::computeForces(CUfunction* kernel_force, const uint gs) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        CHECK(cuLaunchKernel(*kernel_force, gs, 1, 1, bs, 1, 1, 0, nullptr, nullptr, nullptr));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.forceUpdateTime += (elapsedTime * 16);
    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplSlangCuda<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);

        // =============================================================================
        //                               Kernel Parameters
        // =============================================================================

        auto& soa = *_particles;
        CUdeviceptr d_positions    = reinterpret_cast<CUdeviceptr>(soa.positions);
        CUdeviceptr d_velocities   = reinterpret_cast<CUdeviceptr>(soa.velocities);
        CUdeviceptr d_forces       = reinterpret_cast<CUdeviceptr>(soa.forces);
        CUdeviceptr d_oldForces    = reinterpret_cast<CUdeviceptr>(soa.oldForces);

        CUdeviceptr d_cells        = reinterpret_cast<CUdeviceptr>(soa.cells);
        CUdeviceptr d_particleIdx  = reinterpret_cast<CUdeviceptr>(soa.particleIdx);
        CUdeviceptr d_idCells      = reinterpret_cast<CUdeviceptr>(soa.idCells);

        // Parameters for KernelResetCells.ptx
        ResetPushConstants reset_pc_host{};
        reset_pc_host.total_size = soa.cellsLength;
        // copying push constants
        CUdeviceptr reset_pc_ptr;
        CHECK(cuMemAlloc(&reset_pc_ptr, sizeof(ResetPushConstants)));
        CHECK(cuMemcpyHtoD(reset_pc_ptr, &reset_pc_host, sizeof(ResetPushConstants)));

        PushReset params_resetCells{};
        params_resetCells.cells = ResourceSlot{d_cells, 0};
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
        hist_pc_host.bMax_x       = _config.boxMax[0];
        hist_pc_host.bMax_y       = _config.boxMax[1];
        hist_pc_host.bMax_z       = _config.boxMax[2];
        // copying push constants
        CUdeviceptr hist_pc_ptr;
        CHECK(cuMemAlloc(&hist_pc_ptr, sizeof(HistPushConstants)));
        CHECK(cuMemcpyHtoD(hist_pc_ptr, &hist_pc_host, sizeof(HistPushConstants)));

        PushHist params_histogram{};
        params_histogram.positions = ResourceSlot{d_positions, 0};
        params_histogram.histogram = ResourceSlot{d_cells, 0};
        params_histogram.idCells   = ResourceSlot{d_idCells, 0};
        params_histogram.pc        = hist_pc_ptr;

        // Parameters for KernelIdCells.ptx
        IdPushConstants id_pc_host{};
        id_pc_host.numParticles = _config.size;
        // copying push constants
        CUdeviceptr id_pc_ptr;
        CHECK(cuMemAlloc(&id_pc_ptr, sizeof(HistPushConstants)));
        CHECK(cuMemcpyHtoD(id_pc_ptr, &id_pc_host, sizeof(HistPushConstants)));

        PushHist params_idCells{};
        params_idCells.particleIdx = ResourceSlot{d_particleIdx, 0};
        params_idCells.idCells     = ResourceSlot{d_idCells, 0};
        params_idCells.starts      = ResourceSlot{d_cells, 0};
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
        params_position.positions  = ResourceSlot{d_positions, 0};
        params_position.velocities = ResourceSlot{d_velocities, 0};
        params_position.forces     = ResourceSlot{d_forces, 0};
        params_position.oldForces  = ResourceSlot{d_oldForces, 0};
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
        params_velocity.velocities = ResourceSlot{d_velocities, 0};
        params_velocity.forces     = ResourceSlot{d_forces, 0};
        params_velocity.oldForces  = ResourceSlot{d_oldForces, 0};
        params_velocity.pc         = vel_pc_ptr;

        // Parameters for KernelForce.ptx
        ForPushConstants for_pc_host{};
        for_pc_host.numParticles = _config.size;
        // copying push constants
        CUdeviceptr for_pc_ptr;
        CHECK(cuMemAlloc(&for_pc_ptr, sizeof(ForPushConstants)));
        CHECK(cuMemcpyHtoD(for_pc_ptr, &for_pc_host, sizeof(ForPushConstants)));

        PushFor params_force{};
        params_force.positions     = ResourceSlot{d_positions, 0};
        params_force.forces        = ResourceSlot{d_forces, 0};
        params_force.pc            = for_pc_ptr;

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

        CUmodule module_exclusive;
        CUfunction kernel_blellochScan;
        CUfunction kernel_blockSum;
        setupLinkedModule(&module_exclusive, &kernel_blellochScan, &kernel_blockSum, SLANG_PTX_DIR "/KernelBlellochScan.ptx", SLANG_PTX_DIR "/KernelBlockSum.ptx", "computeBlellochScan", "computeBlockSum", "Params_Exclusive");
        CUdeviceptr memory_exclusive;
        size_t size_exclusive;
        CHECK(cuModuleGetGlobal(
            &memory_exclusive,
            &size_exclusive,
            *module_exclusive,
            "Params_ExclusiveScan"
        ));

        const uint32_t _gridSizePerParticle = util::ceilDiv<unsigned int>(size, _blockSize);
        const uint32_t _gridSizePerCell     = util::ceilDiv<unsigned int>(soa.cellsLength, _blockSize);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            computeResetCells(&kernel_resetCells, _gridSizePerCell);
            computeHistogram(&kernel_histogram, _gridSizePerParticle);
            computeBlellochScan(&d_cells, exclusiveScan_memory, kernel_blellochScan, kernel_blockSum, soa.cellsLength);
            computeIdCells(&kernel_idCells, _gridSizePerParticle,);

            updatePositionsAndResetForce(&kernel_position, _gridSizePerParticle);
            computeForces(&kernel_force, _gridSizePerParticle);
            updateVelocities(&kernel_velocity, _gridSizePerParticle);
        }
        //_particles->print_buffer(soa.positions, _config.size);

        freeData(reset_pc_ptr, module_resetCells);
        freeData(hist_pc_ptr, module_histogram);
        freeData(id_pc_ptr, module_idCells);
        freeData(pos_pc_ptr, module_position);
        freeData(vel_pc_ptr, module_velocity);
        freeData(for_pc_ptr, module_force);
        CHECK(cuModuleUnload(*module_exclusive));
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplSlangCuda<float>;

};
