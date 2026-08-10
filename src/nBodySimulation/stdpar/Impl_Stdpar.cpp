#include "Impl_Stdpar.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>

namespace ppb {

    template <typename FloatType>
    StdparParticleSoA<FloatType>::StdparParticleSoA(const std::vector<Particle<FloatType>> &ref)
        : _ref{ref}
        , positions(ref.size() * 3, 0.0)
        , velocities(ref.size() * 3, 0.0)
        , forces(ref.size() * 3, 0.0)
        , oldForces(ref.size() * 3, 0.0)
        , indices(ref.size())
    {
        std::iota(indices.begin(), indices.end(), size_t{0});
        for (size_t i = 0; i < ref.size() * 3; ++i) {
            const size_t particleIndex = i / 3;
            const size_t componentIndex = i % 3;
            positions[i] = ref[particleIndex].getPosition()[componentIndex];
            velocities[i] = ref[particleIndex].getVelocity()[componentIndex];
            forces[i] = ref[particleIndex].getForce()[componentIndex];
        }
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> StdparParticleSoA<FloatType>::toParticles() const {
        std::vector<Particle<FloatType>> particles{_ref};
        for (size_t i = 0; i < _ref.size(); ++i) {
            std::array<FloatType, 3> pos{};
            std::array<FloatType, 3> vel{};
            std::array<FloatType, 3> force{};
            for (size_t j = 0; j < 3; ++j) {
                pos[j] = positions[i * 3 + j];
                vel[j] = velocities[i * 3 + j];
                force[j] = forces[i * 3 + j];
            }
            particles[i].setPosition(pos);
            particles[i].setVelocity(vel);
            particles[i].setForce(force);
        }
        return particles;
    }


    template <typename FloatType>
    ImplStdpar<FloatType>::ImplStdpar(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplStdpar<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
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
    void ImplStdpar<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        const FloatType dt = _config.deltaT;
        const std::array<FloatType, 3> globalForce = _config.globalForce;
        FloatType *forces = _particles->forces.data();
        FloatType *oldForces = _particles->oldForces.data();
        const FloatType *velocities = _particles->velocities.data();
        FloatType *positions = _particles->positions.data();

        constexpr FloatType m = 1.0;
        const FloatType tt2m = (dt * dt / (2 * m));

        const size_t *indices = _particles->indices.data();

        const auto start = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par_unseq, indices, indices + size, [=](const size_t i) {
            for (size_t j = 0; j < 3; ++j) {
                const size_t index = i * 3 + j;
                const auto v = velocities[index];
                const auto f = forces[index];
                oldForces[index] = f;
                forces[index] = globalForce[j];
                positions[index] += (v * dt + f * tt2m);
            }
        });
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.positionUpdateForceResetTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template <typename FloatType>
    void ImplStdpar<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        const FloatType dt = _config.deltaT;
        const FloatType *forces = _particles->forces.data();
        const FloatType *oldForces = _particles->oldForces.data();
        FloatType *velocities = _particles->velocities.data();

        constexpr FloatType m = 1.0;
        const FloatType t2m = (dt / (2 * m));

        const size_t *indices = _particles->indices.data();

        const auto start = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par_unseq, indices, indices + size, [=](const size_t i) {
            for (size_t j = 0; j < 3; ++j) {
                const size_t index = i * 3 + j;
                velocities[index] += ((forces[index] + oldForces[index]) * t2m);
            }
        });
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.velocityUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template <typename FloatType>
    void ImplStdpar<FloatType>::computeForces() {
        const size_t size = _config.size;
        FloatType *forces = _particles->forces.data();
        const FloatType *positions = _particles->positions.data();

        const size_t *indices = _particles->indices.data();

        const auto start = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par_unseq, indices, indices + size, [=](const size_t i) {
            constexpr FloatType sigma = (1.0 + 1.0) * 0.5;
            constexpr FloatType sigmaSquared = sigma * sigma;
            const FloatType epsilon24 = std::sqrt(FloatType{1.0} * FloatType{1.0}) * 24;

            FloatType acc[3] = {0.0, 0.0, 0.0};
            for (size_t j = 0; j < size; ++j) {
                if (i == j) {
                    continue;
                }

                FloatType dr[3];
                FloatType dr2 = 0.0;
                for (size_t d = 0; d < 3; ++d) {
                    dr[d] = positions[i * 3 + d] - positions[j * 3 + d];
                    dr2 += dr[d] * dr[d];
                }
                const auto invdr2 = 1. / dr2;
                auto lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const auto lj12 = lj6 * lj6;
                const auto lj12m6 = lj12 - lj6;
                const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                for (size_t d = 0; d < 3; ++d) {
                    acc[d] += dr[d] * fac;
                }
            }
            for (size_t d = 0; d < 3; ++d) {
                forces[i * 3 + d] += acc[d];
            }
        });
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.forceUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template class ImplStdpar<float>;
    template class StdparParticleSoA<float>;

    template class ImplStdpar<double>;
    template class StdparParticleSoA<double>;

} // namespace ppb
