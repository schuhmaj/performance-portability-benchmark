#include "Impl_Raja.h"

#include <chrono>
#include "RAJA/RAJA.hpp"

namespace ppb {

    namespace {
#if defined(RAJA_ENABLE_CUDA)
        using ExecPolicy = RAJA::cuda_exec<256>;
        using Resource = RAJA::resources::Cuda;
#elif defined(RAJA_ENABLE_HIP)
        using ExecPolicy = RAJA::hip_exec<256>;
        using Resource = RAJA::resources::Hip;
#elif defined(RAJA_ENABLE_SYCL)
        using ExecPolicy = RAJA::sycl_exec<256>;
        using Resource = RAJA::resources::Sycl;
#elif defined(RAJA_ENABLE_OPENMP)
        using ExecPolicy = RAJA::omp_parallel_for_exec;
        using Resource = RAJA::resources::Host;
#else
        using ExecPolicy = RAJA::seq_exec;
        using Resource = RAJA::resources::Host;
#endif

        double elapsedNanoseconds(const std::chrono::steady_clock::time_point &start) {
            return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        }
    }

    template <typename FloatType>
    RajaParticleSoA<FloatType>::RajaParticleSoA(const std::vector<Particle<FloatType>> &particles)
        : _ref{particles} {
        const size_t n = particles.size() * 3;
        Resource res = Resource::get_default();
        positions = res.template allocate<FloatType>(n);
        velocities = res.template allocate<FloatType>(n);
        forces = res.template allocate<FloatType>(n);
        oldForces = res.template allocate<FloatType>(n);

        std::vector<FloatType> hostPositions(n);
        std::vector<FloatType> hostVelocities(n);
        std::vector<FloatType> hostForces(n);
        for (size_t i = 0; i < particles.size(); ++i) {
            for (size_t j = 0; j < 3; ++j) {
                hostPositions[i * 3 + j] = particles[i].getPosition()[j];
                hostVelocities[i * 3 + j] = particles[i].getVelocity()[j];
                hostForces[i * 3 + j] = particles[i].getForce()[j];
            }
        }
        res.memcpy(positions, hostPositions.data(), n * sizeof(FloatType));
        res.memcpy(velocities, hostVelocities.data(), n * sizeof(FloatType));
        res.memcpy(forces, hostForces.data(), n * sizeof(FloatType));
        res.wait();
    }

    template <typename FloatType>
    RajaParticleSoA<FloatType>::~RajaParticleSoA() {
        Resource res = Resource::get_default();
        res.deallocate(positions);
        res.deallocate(velocities);
        res.deallocate(forces);
        res.deallocate(oldForces);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> RajaParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        const size_t n = particles.size() * 3;
        Resource res = Resource::get_default();

        std::vector<FloatType> hostPositions(n);
        std::vector<FloatType> hostVelocities(n);
        std::vector<FloatType> hostForces(n);
        res.memcpy(hostPositions.data(), positions, n * sizeof(FloatType));
        res.memcpy(hostVelocities.data(), velocities, n * sizeof(FloatType));
        res.memcpy(hostForces.data(), forces, n * sizeof(FloatType));
        res.wait();

        for (size_t i = 0; i < particles.size(); ++i) {
            std::array<FloatType, 3> pos{};
            std::array<FloatType, 3> vel{};
            std::array<FloatType, 3> force{};
            for (size_t j = 0; j < 3; ++j) {
                pos[j] = hostPositions[i * 3 + j];
                vel[j] = hostVelocities[i * 3 + j];
                force[j] = hostForces[i * 3 + j];
            }
            particles[i].setPosition(pos);
            particles[i].setVelocity(vel);
            particles[i].setForce(force);
        }
        return particles;
    }

    template <typename FloatType>
    size_t RajaParticleSoA<FloatType>::size() const {
        return _ref.size();
    }


    template class RajaParticleSoA<float>;
    template class RajaParticleSoA<double>;



    template<typename FloatType>
    ImplRaja<FloatType>::ImplRaja(const ParticleSimulationConfig<FloatType> &config) : _config{config} {

    }

    template<typename FloatType>
    void ImplRaja<FloatType>::updatePositionsAndResetForce() {
        const size_t n = _particles->size() * 3;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        const auto globalForce0 = static_cast<FloatType>(_config.globalForce[0]);
        const auto globalForce1 = static_cast<FloatType>(_config.globalForce[1]);
        const auto globalForce2 = static_cast<FloatType>(_config.globalForce[2]);
        FloatType *force = _particles->forces;
        FloatType *oldForce = _particles->oldForces;
        FloatType *velocity = _particles->velocities;
        FloatType *position = _particles->positions;

        Resource res = Resource::get_default();
        const auto start = std::chrono::steady_clock::now();
        RAJA::forall<ExecPolicy>(res, RAJA::TypedRangeSegment<size_t>(0, n), [=] RAJA_HOST_DEVICE(const size_t idx) {
            const size_t j = idx % 3;
            constexpr FloatType m = 1.0;
            auto v = velocity[idx];
            auto f = force[idx];

            oldForce[idx] = f;
            force[idx] = j == 0 ? globalForce0 : (j == 1 ? globalForce1 : globalForce2);

            v *= dt;
            f *= (dt * dt / (2 * m));
            const auto displacement = v + f;
            position[idx] = position[idx] + displacement;
        });
        res.wait();
        _timings.positionUpdateForceResetTime += elapsedNanoseconds(start);
    }

    template<typename FloatType>
    void ImplRaja<FloatType>::updateVelocities() {
        const size_t n = _particles->size() * 3;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        FloatType *force = _particles->forces;
        FloatType *oldForce = _particles->oldForces;
        FloatType *velocity = _particles->velocities;

        Resource res = Resource::get_default();
        const auto start = std::chrono::steady_clock::now();
        RAJA::forall<ExecPolicy>(res, RAJA::TypedRangeSegment<size_t>(0, n), [=] RAJA_HOST_DEVICE(const size_t idx) {
            constexpr FloatType m = 1.0;
            const auto changeInVel = (force[idx] + oldForce[idx]) * (dt / (2 * m));
            velocity[idx] = velocity[idx] + changeInVel;
        });
        res.wait();
        _timings.velocityUpdateTime += elapsedNanoseconds(start);
    }

    template<typename FloatType>
    void ImplRaja<FloatType>::computeForces() {
        const size_t size = _particles->size();
        FloatType *force = _particles->forces;
        FloatType *position = _particles->positions;

        Resource res = Resource::get_default();
        const auto start = std::chrono::steady_clock::now();
        RAJA::forall<ExecPolicy>(res, RAJA::TypedRangeSegment<size_t>(0, size), [=] RAJA_HOST_DEVICE(const size_t i) {
            constexpr FloatType sigmaSrc = 1.0;
            constexpr FloatType epsilonSrc = 1.0;
            constexpr FloatType sigma = (sigmaSrc + sigmaSrc) * 0.5;
            constexpr FloatType sigmaSquared = sigma * sigma;
            const FloatType epsilon24 = std::sqrt(epsilonSrc * epsilonSrc) * 24.0;

            FloatType acc[3] = {0, 0, 0};
            for (size_t j = 0; j < size; ++j) {
                if (i == j) {
                    continue;
                }
                FloatType dr[3];
                FloatType dr2 = 0;
                for (size_t k = 0; k < 3; ++k) {
                    dr[k] = position[i * 3 + k] - position[j * 3 + k];
                    dr2 += dr[k] * dr[k];
                }

                const FloatType invdr2 = 1.0 / dr2;
                FloatType lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const FloatType lj12 = lj6 * lj6;
                const FloatType lj12m6 = lj12 - lj6;
                const FloatType fac = epsilon24 * (lj12 + lj12m6) * invdr2;

                for (size_t k = 0; k < 3; ++k) {
                    acc[k] += dr[k] * fac;
                }
            }

            for (size_t k = 0; k < 3; ++k) {
                force[i * 3 + k] += acc[k];
            }
        });
        res.wait();
        _timings.forceUpdateTime += elapsedNanoseconds(start);
    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplRaja<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }

        return std::make_pair(_particles->toParticles(), _timings);
    }

    /* Explicit Instantiation for float and double */
    template class ImplRaja<float>;
    template class ImplRaja<double>;

};
