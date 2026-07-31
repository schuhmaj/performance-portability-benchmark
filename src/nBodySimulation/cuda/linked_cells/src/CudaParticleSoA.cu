#include "CudaParticleSoA.cuh"
#include "common.cuh"

#define CHECK_CUDA_ERROR(val) ppb::cuda::nbody::check((val), #val, __FILE__, __LINE__)

namespace ppb::cuda::nbody {
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

        CHECK_CUDA_ERROR(cudaMalloc(&positions, sizeof(float4) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&velocities, sizeof(float4) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&forces, sizeof(float4) * size));
        CHECK_CUDA_ERROR(cudaMalloc(&oldForces, sizeof(float4) * size));

        CHECK_CUDA_ERROR(cudaMemcpy(positions, positionsHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(velocities, velocitiesHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(forces, forcesHost.data(), sizeof(float4) * size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(oldForces, 0.0, sizeof(float4) * size));
    }

    template <typename FloatType>
    CudaParticleSoA<FloatType>::~CudaParticleSoA() {
        CHECK_CUDA_ERROR(cudaFree(positions));
        CHECK_CUDA_ERROR(cudaFree(velocities));
        CHECK_CUDA_ERROR(cudaFree(forces));
        CHECK_CUDA_ERROR(cudaFree(oldForces));
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> CudaParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        CHECK_CUDA_ERROR(cudaMemcpy(positionsHost.data(), positions, sizeof(float4) * _ref.size(), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(velocitiesHost.data(), velocities, sizeof(float4) * _ref.size(), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(forcesHost.data(), forces, sizeof(float4) * _ref.size(), cudaMemcpyDeviceToHost));
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

    template class CudaParticleSoA<float>;
} // namespace ppb::cuda::nbody