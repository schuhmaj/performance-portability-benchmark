#include "Impl_Slang_Cuda.cuh"
#include "Kernel_Structs.cuh"
#include <cuda_runtime.h>

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
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) : _config{config}, _globalForce{_config.globalForce[0], _config.globalForce[1], _config.globalForce[2]} {
        const size_t size = _config.size;
        constexpr unsigned int WRAP_SIZE = _config.TILE_SIZE;
        constexpr unsigned int MAX_THREADS = 1024;

        if (size <= MAX_THREADS) {
            _blockSize = size;
        } else {
            int blockSize = 0;
            int minGridSize = 0;
        }
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize);
    }

    template <typename FloatType>
    void ImplCuda<FloatType>::setupKernel(void* pushData, CUmodule* module_, CUfunction* kernel, const char* file, const char* name, const char* params, size_t pushSize) {
        cuModuleLoad(module_, file);
        cuModuleGetFunction(kernel, *module_, name);
        CUdeviceptr memory;
        size_t size;
        cuModuleGetGlobal(
            &memory,
            &size,
            *module_,
            params
        );
        cuMemcpyHtoD(memory, pushData, pushSize);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updatePositionsAndResetForce(CUfunction* kernel_position) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cuLaunchKernel(*kernel_position, _gridSize, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.positionUpdateForceResetTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updateVelocities(CUfunction* kernel_velocity) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cuLaunchKernel(*kernel_velocity, _gridSize, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        _timings.velocityUpdateTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::computeForces(CUfunction* kernel_force) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cuLaunchKernel(*kernel_force, _gridSize, 1, 1, _blockSize, 1, 1, 0, nullptr, nullptr, nullptr);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.forceUpdateTime += (elapsedTime * 16);
    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplCuda<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);
        auto& soa = *_particles;

        PushPos params_position{};
        params_position.positions     = soa.positions;
        params_position.velocities    = soa.velocities;
        params_position.forces        = soa.forces;
        params_position.oldForces     = soa.oldForces;
        params_position.globalForce_x = _config.globalForce[0];
        params_position.globalForce_y = _config.globalForce[1];
        params_position.globalForce_z = _config.globalForce[2];
        params_position.dt            = _config.deltaT;
        params_position.numParticles  = _config.size;

        PushVel params_velocity{};
        params_velocity.velocities    = soa.velocities;
        params_velocity.forces        = soa.forces;
        params_velocity.oldForces     = soa.oldForces;
        params_velocity.dt            = _config.deltaT;
        params_velocity.numParticles  = _config.size;

        PushFor params_force{};
        params_force.positions        = soa.positions;
        params_force.forces           = soa.forces;
        params_force.numParticles     = _config.size;

        CUmodule module_position;
        CUfunction kernel_position;
        setupKernel(&params_position, &module_position, &kernel_position, "generated_shaders/KernelPosition.ptx", "computePosition", "Params_Position", sizeof(PushPos));

        CUmodule module_velocity;
        CUfunction kernel_velocity;
        setupKernel(&params_velocity, &module_velocity, &kernel_velocity, "generated_shaders/KernelVelocity.ptx", "computeVelocity", "Params_Velocity", sizeof(PushVel));

        CUmodule module_force;
        CUfunction kernel_force;
        setupKernel(&params_force, &module_force, &kernel_force, "generated_shaders/KernelForce.ptx", "computeForce", "Params_Force", sizeof(PushFor));

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce(&kernel_position);
            computeForces(&kernel_velocity);
            updateVelocities(&kernel_force);
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template class ImplCuda<float>;

};