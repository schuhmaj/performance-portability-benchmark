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

        CUdevice device;
        CHECK(cuDeviceGet(&device, 0));
        CUcontext context;
        CHECK(cuCtxCreate(&context, 0, device));

        CHECK(cuMemAlloc(&positions, sizeof(float4) * size));
        CHECK(cuMemAlloc(&velocities, sizeof(float4) * size));
        CHECK(cuMemAlloc(&forces, sizeof(float4) * size));
        CHECK(cuMemAlloc(&oldForces, sizeof(float4) * size));

        CHECK(cuMemcpyHtoD(positions, positionsHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(velocities, velocitiesHost.data(), sizeof(float4) * size));
        CHECK(cuMemcpyHtoD(forces, forcesHost.data(), sizeof(float4) * size));
        CHECK(cuMemsetD8(oldForces, 0, sizeof(float4) * size));

    }   

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        CHECK(cuMemFree(positions));
        CHECK(cuMemFree(velocities));
        CHECK(cuMemFree(forces));
        CHECK(cuMemFree(oldForces));
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

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplSlangCuda<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);

        // =============================================================================
        //                               Kernel Parameters
        // =============================================================================

        auto& soa = *_particles;

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
        params_force.positions     = ResourceSlot{soa.positions, 0};
        params_force.forces        = ResourceSlot{soa.forces, 0};
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

        const uint32_t _gridSize = util::ceilDiv<unsigned int>(_config.size, _blockSize);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            launchKernel(&kernel_position, _gridSize, &_timings.positionUpdateForceResetTime);
            launchKernel(&kernel_force, _gridSize, &_timings.forceUpdateTime);
            launchKernel(&kernel_velocity, _gridSize, &_timings.velocityUpdateTime);
        }
        _particles->print_buffer(soa.positions, _config.size);

        freeData(pos_pc_ptr, &module_position);
        freeData(vel_pc_ptr, &module_velocity);
        freeData(for_pc_ptr, &module_force);
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplSlangCuda<float>;

};