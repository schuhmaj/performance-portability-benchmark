#include "Impl_OpenMP.h"

namespace ppb {

    template <typename FloatType>
    OpenMPParticleSoA<FloatType>::OpenMPParticleSoA(const std::vector<Particle<FloatType>> &ref)
        : _ref{ref}
        , positionsHost(ref.size() * 3, 0.0)
        , velocitiesHost(ref.size() * 3, 0.0)
        , forcesHost(ref.size() * 3, 0.0)
    {
        const size_t numberOfBytes = ref.size() * sizeof(FloatType) * 3;
        positions = static_cast<FloatType *>(omp_target_alloc(numberOfBytes, DEVICE_ID));
        velocities = static_cast<FloatType *>(omp_target_alloc(numberOfBytes, DEVICE_ID));
        forces = static_cast<FloatType *>(omp_target_alloc(numberOfBytes, DEVICE_ID));
        oldForces = static_cast<FloatType *>(omp_target_alloc(numberOfBytes, DEVICE_ID));
        for (size_t i = 0; i < ref.size() * 3; ++i) {
            const size_t particleIndex = i / 3;
            const size_t componentIndex = i % 3;
            positionsHost[i] = ref[particleIndex].getPosition()[componentIndex];
            velocitiesHost[i] = ref[particleIndex].getVelocity()[componentIndex];
            forcesHost[i] = ref[particleIndex].getForce()[componentIndex];
        }
        omp_target_memcpy(positions, positionsHost.data(), numberOfBytes, 0, 0, DEVICE_ID, HOST_ID);
        omp_target_memcpy(velocities, velocitiesHost.data(), numberOfBytes, 0, 0, DEVICE_ID, HOST_ID);
        omp_target_memcpy(forces, forcesHost.data(), numberOfBytes, 0, 0, DEVICE_ID, HOST_ID);
#if defined(_OPENMP) && _OPENMP >= 202411
        // omp_target_memset was introduced in OpenMP 6.0 (_OPENMP == 202411).
        omp_target_memset(oldForces, 0, numberOfBytes, DEVICE_ID);
#else
        // Fallback for older compilers (e.g. oneAPI 2023): zero the device buffer by
        // copying from a zero-filled host buffer.
        const std::vector<FloatType> zeros(ref.size() * 3, 0.0);
        omp_target_memcpy(oldForces, zeros.data(), numberOfBytes, 0, 0, DEVICE_ID, HOST_ID);
#endif
    }

    template <typename FloatType>
    OpenMPParticleSoA<FloatType>::~OpenMPParticleSoA() {
        omp_target_free(positions, DEVICE_ID);
        omp_target_free(velocities, DEVICE_ID);
        omp_target_free(forces, DEVICE_ID);
        omp_target_free(oldForces, DEVICE_ID);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> OpenMPParticleSoA<FloatType>::toParticles() {
        const size_t numberOfBytes = _ref.size() * sizeof(FloatType) * 3;
        std::vector<Particle<FloatType>> particles{_ref};
        omp_target_memcpy(positionsHost.data(), positions, numberOfBytes, 0, 0, HOST_ID, DEVICE_ID);
        omp_target_memcpy(velocitiesHost.data(), velocities, numberOfBytes, 0, 0, HOST_ID, DEVICE_ID);
        omp_target_memcpy(forcesHost.data(), forces, numberOfBytes, 0, 0, HOST_ID, DEVICE_ID);
        for (size_t i = 0; i < _ref.size(); ++i) {
            std::array<FloatType, 3> pos{};
            std::array<FloatType, 3> vel{};
            std::array<FloatType, 3> force{};
            for (size_t j = 0; j < 3; ++j) {
                pos[j] = positionsHost[i * 3 + j];
                vel[j] = velocitiesHost[i * 3 + j];
                force[j] = forcesHost[i * 3 + j];
            }
            particles[i].setPosition(pos);
            particles[i].setVelocity(vel);
            particles[i].setForce(force);
        }
        return particles;
    }


    template <typename FloatType>
    ImplOpenMP<FloatType>::ImplOpenMP(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplOpenMP<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _particles.emplace(particles);
        _timings.reset();
        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template <typename FloatType>
    void ImplOpenMP<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        const FloatType dt = _config.deltaT;
        const std::array<float_type, 3> &globalForce = _config.globalForce;
        auto &forces = _particles->forces;
        auto &oldForces = _particles->oldForces;
        const auto &velocities = _particles->velocities;
        auto &positions = _particles->positions;

        constexpr auto m = 1.0;
        const FloatType tt2m = (_config.deltaT * _config.deltaT / (2 * m));

        const auto start = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                const size_t index = i * 3 + j;
                auto v = velocities[index];
                auto f = forces[index];
                oldForces[index] = f;
                forces[index] = globalForce[j];
                positions[index] += (v * dt + f  * tt2m);
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.positionUpdateForceResetTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    template <typename FloatType>
    void ImplOpenMP<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        const FloatType dt = _config.deltaT;
        const std::array<float_type, 3> &globalForce = _config.globalForce;
        const auto &forces = _particles->forces;
        const auto &oldForces = _particles->oldForces;
        auto &velocities = _particles->velocities;

        constexpr FloatType m = 1.0;
        const FloatType t2m = (dt / (2.0 * m));

        const auto start = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                const size_t index = i * 3 + j;
                velocities[index] += ((forces[index] + oldForces[index]) * t2m);
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.velocityUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template <typename FloatType>
    void ImplOpenMP<FloatType>::computeForces() {
        const size_t size = _config.size;
        const FloatType dt = _config.deltaT;
        const std::array<float_type, 3> &globalForce = _config.globalForce;
        auto &forces = _particles->forces;
        auto &positions = _particles->positions;

        constexpr FloatType m = 1.0;
        const FloatType t2m = (dt / (2.0 * m));

        const auto start = std::chrono::high_resolution_clock::now();
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (i == j) {
                    continue;
                }

                constexpr FloatType sigma = (1.0 + 1.0) * 0.5;
                constexpr FloatType sigmaSquared = sigma * sigma;
                const FloatType epsilon24 = std::sqrt(1.0 * 1.0) * 24;

                FloatType dr[3];
                FloatType dr2 = 0.0;
                for (size_t d = 0; d < 3; ++d) {
                    const size_t indexI = i * 3 + d;
                    const size_t indexJ = j * 3 + d;
                    dr[d] = positions[indexI] - positions[indexJ];
                    dr2 += dr[d] * dr[d];
                }
                const auto invdr2 = 1. / dr2;
                auto lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const auto lj12 = lj6 * lj6;
                const auto lj12m6 = lj12 - lj6;
                const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                for (size_t d = 0; d < 3; ++d) {
                    const auto f = dr[d] * fac;
                    forces[i * 3 + d] += f;
                }
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.forceUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::operator-(end, start)).count());
    }

    template class ImplOpenMP<float>;
    template class OpenMPParticleSoA<float>;

    template class ImplOpenMP<double>;
    template class OpenMPParticleSoA<double>;

} // namespace ppb


