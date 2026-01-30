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

        CHECK(cuInit(0));

        cudaMalloc(&positions, sizeof(float4) * size);
        cudaMalloc(&velocities, sizeof(float4) * size);
        cudaMalloc(&forces, sizeof(float4) * size);
        cudaMalloc(&oldForces, sizeof(float4) * size);

        cudaMemcpy(positions, positionsHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(velocities, velocitiesHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(forces, forcesHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice);
        cudaMemset(oldForces, 0.0, sizeof(float4) * size);
    }   

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        cudaFree(positions);
        cudaFree(velocities);
        cudaFree(forces);
        cudaFree(oldForces);
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
        CHECK(cuLaunchKernel(*kernel_force, gs, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr));
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
        CUdeviceptr d_positions  = reinterpret_cast<CUdeviceptr>(soa.positions);
        CUdeviceptr d_velocities = reinterpret_cast<CUdeviceptr>(soa.velocities);
        CUdeviceptr d_forces     = reinterpret_cast<CUdeviceptr>(soa.forces);
        CUdeviceptr d_oldForces  = reinterpret_cast<CUdeviceptr>(soa.oldForces);

        // Parameters for KernelPosition.ptx
        PosPushConstants pos_pc_host{};
        pos_pc_host.globalForce_x = _config.globalForce[0];
        pos_pc_host.globalForce_y = _config.globalForce[1];
        pos_pc_host.globalForce_z = _config.globalForce[2];
        pos_pc_host.dt = _config.deltaT;
        pos_pc_host.numParticles = _config.size;
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

        CUmodule module_position;
        CUfunction kernel_position;
        setupKernel(&params_position, &module_position, &kernel_position, SLANG_PTX_DIR "/KernelPosition.ptx", "computePosition", "Params_Position", sizeof(PushPos));

        CUmodule module_velocity;
        CUfunction kernel_velocity;
        setupKernel(&params_velocity, &module_velocity, &kernel_velocity, SLANG_PTX_DIR "/KernelVelocity.ptx", "computeVelocity", "Params_Velocity", sizeof(PushVel));

        CUmodule module_force;
        CUfunction kernel_force;
        setupKernel(&params_force, &module_force, &kernel_force, SLANG_PTX_DIR "/KernelForce.ptx", "computeForce", "Params_Force", sizeof(PushFor));

        const uint32_t _gridSize = util::ceilDiv<unsigned int>(size, _blockSize);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce(&kernel_position, _gridSize);
            computeForces(&kernel_force, _gridSize);
            updateVelocities(&kernel_velocity, _gridSize);
        }
        //_particles->print_buffer(soa.positions, _config.size);

        freeData(pos_pc_ptr, module_position);
        freeData(vel_pc_ptr, module_velocity);
        freeData(for_pc_ptr, module_force);
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplSlangCuda<float>;

};