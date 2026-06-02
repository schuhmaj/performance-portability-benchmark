#include "Impl_Cuda.cuh"
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
            positionsHost[i] = {particles[i].getPosition()[0], particles[i].getPosition()[1], particles[i].getPosition()[2]};
            velocitiesHost[i] = {particles[i].getVelocity()[0], particles[i].getVelocity()[1], particles[i].getVelocity()[2]};
            forcesHost[i] = {particles[i].getForce()[0], particles[i].getForce()[1], particles[i].getForce()[2]};
        }
        cudaMalloc(&positions, sizeof(VectorType3) * size);
        cudaMalloc(&velocities, sizeof(VectorType3) * size);
        cudaMalloc(&forces, sizeof(VectorType3) * size);
        cudaMalloc(&oldForces, sizeof(VectorType3) * size);

        cudaMemcpy(positions, positionsHost.data(), sizeof(VectorType3) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(velocities, velocitiesHost.data(), sizeof(VectorType3) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(forces, forcesHost.data(), sizeof(VectorType3) * size, cudaMemcpyHostToDevice);
        cudaMemset(oldForces, 0.0, sizeof(VectorType3) * size);
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
        cudaMemcpy(positionsHost.data(), positions, sizeof(VectorType3) * _ref.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(velocitiesHost.data(), velocities, sizeof(VectorType3) * _ref.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(forcesHost.data(), forces, sizeof(VectorType3) * _ref.size(), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < particles.size(); ++i) {
            const VectorType3& position = positionsHost[i];
            const VectorType3& velocity = velocitiesHost[i];
            const VectorType3& force = forcesHost[i];
            particles[i].setPosition({position.x, position.y, position.z});
            particles[i].setVelocity({velocity.x, velocity.y, velocity.z});
            particles[i].setForce({force.x, force.y, force.z});
        }
        return particles;
    }

    template class CudaParticleSoA<float>;
    template class CudaParticleSoA<double>;

    __global__ void update_positions(VectorType3* positions, const VectorType3* velocities, VectorType3* forces, VectorType3* oldForces, const VectorType3 globalForce, const float deltaT, const size_t numParticles) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }
        constexpr float_type mass = 1.0;
        const VectorType3 force = forces[i];
        const VectorType3 velocity = velocities[i];
        oldForces[i] = force;
        forces[i] = globalForce;

        const VectorType3 velocityPart = {velocity.x * deltaT, velocity.y * deltaT, velocity.z * deltaT};
        const float_type tt2m = deltaT * deltaT / (2.0f * mass);
        const VectorType3 forcePart = {force.x * tt2m, force.y * tt2m, force.z * tt2m};
        const VectorType3 displacement = {velocityPart.x + forcePart.x, velocityPart.y + forcePart.y, velocityPart.z + forcePart.z};
        positions[i] = {positions[i].x + displacement.x, positions[i].y + displacement.y, positions[i].z + displacement.z};
    }

    __global__ void update_velocities(VectorType3* velocities, const VectorType3* forces, const VectorType3* oldForces, const float_type deltaT, const size_t numParticles) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        constexpr float_type mass = 1.0;
        const VectorType3 force = forces[i];
        const VectorType3 oldForce = oldForces[i];
        const VectorType3 velocity = velocities[i];

        const VectorType3 forcePart = {force.x + oldForce.x, force.y + oldForce.y, force.z + oldForce.z};
        const float_type t2m =  deltaT / (2.0f * mass);
        const VectorType3 velChange = {forcePart.x * t2m, forcePart.y * t2m, forcePart.z * t2m};
        velocities[i] = {velocity.x + velChange.x, velocity.y + velChange.y, velocity.z + velChange.z};
    }

    __device__ inline VectorType3 make_float3_add(const VectorType3 a, const VectorType3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ inline VectorType3 make_float3_sub(const VectorType3 a, const VectorType3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ inline VectorType3 make_float3_scale(const VectorType3 v, const float_type s) {
        return make_float3(v.x * s, v.y * s, v.z * s);
    }

    __device__ inline float_type dot3(const VectorType3 a, const VectorType3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __global__ void compute_forces(
        const VectorType3* __restrict__ positions,
        VectorType3* __restrict__ forces,
        const unsigned int numParticles
    ) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) {
            return;
        }

        VectorType3 fi = make_float3(0.f, 0.f, 0.f);
        for (unsigned int j = 0; j < numParticles; ++j) {
            if (i == j) continue;

            const float_type sigma = 1.0f;
            const float_type sigmaSquared = sigma * sigma;
            const float_type epsilon24 = 24.0f; // 1.0 * 24.0

            const VectorType3 dr = make_float3_sub(positions[i], positions[j]);
            const float_type dr2 = dot3(dr, dr);

            const float_type invdr2 = 1.0f / dr2;
            float_type lj6 = sigmaSquared * invdr2;
            lj6 = lj6 * lj6 * lj6;
            const float_type lj12 = lj6 * lj6;
            const float_type lj12m6 = lj12 - lj6;
            const float_type fac = epsilon24 * (lj12 + lj12m6) * invdr2;

            const VectorType3 f = make_float3_scale(dr, fac);
            fi = make_float3_add(fi, f);
        }
        forces[i] = fi;
    }


    template<typename FloatType>
    ImplCuda<FloatType>::ImplCuda(const ParticleSimulationConfig<FloatType> &config) : _config{config}, _globalForce{_config.globalForce[0], _config.globalForce[1], _config.globalForce[2]} {
        const size_t size = _config.size;
        constexpr unsigned int WRAP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;

        if (size <= MAX_THREADS) {
            _blockSize = size;
        } else {
            int blockSize = 0;
            int minGridSize = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, reinterpret_cast<void *>(update_positions), 0, size);
        }
        _gridSize = util::ceilDiv<unsigned int>(size, _blockSize);
    }



    template<typename FloatType>
    void ImplCuda<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        const auto &globalForce = _config.globalForce;
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        update_positions<<<_gridSize, _blockSize>>>(position, velocity, force, oldForce, _globalForce, dt, size);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.positionUpdateForceResetTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        constexpr size_t dim = 3;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        update_velocities<<<_gridSize, _blockSize>>>(velocity, force, oldForce, dt, size);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        _timings.velocityUpdateTime += (elapsedTime * 1e6);
    }

    template<typename FloatType>
    void ImplCuda<FloatType>::computeForces() {
        const size_t size = _config.size;
        auto &force = _particles->forces;
        auto &position = _particles->positions;

        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        compute_forces<<<_gridSize, _blockSize>>>(position, force, size);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        _timings.forceUpdateTime += (elapsedTime * 16);
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
    template class ImplCuda<double>;

};