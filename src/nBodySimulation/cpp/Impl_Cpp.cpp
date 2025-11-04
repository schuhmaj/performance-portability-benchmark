#include "Impl_Cpp.h"

namespace ppb {

    template <typename FloatType>
    ImplCpp<FloatType>::ImplCpp(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplCpp<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<Particle<FloatType>> particlesCopy{particles};
        _timings.reset();
        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce(particlesCopy);
            computeForces(particlesCopy);
            updateVelocities(particlesCopy);
        }
        return std::make_pair(particlesCopy, _timings);
    }

    template <typename FloatType>
    void ImplCpp<FloatType>::updatePositionsAndResetForce(std::vector<Particle<FloatType>> &particles) {
        using ppb::util::operator+, ppb::util::operator*=;
        const auto start = std::chrono::high_resolution_clock::now();
        for (auto &particle : particles) {
            const auto m = particle.getMass();
            auto v = particle.getVelocity();
            auto f = particle.getForce();
            particle.setOldForce(f);
            particle.setForce(_config.globalForce);
            v *= _config.deltaT;
            f *= (_config.deltaT * _config.deltaT / (2 * m));
            const auto displacement = v + f;
            particle.addPosition(displacement);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.positionUpdateForceResetTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    template <typename FloatType>
    void ImplCpp<FloatType>::updateVelocities(std::vector<Particle<FloatType>> &particles) {
        using ppb::util::operator+, ppb::util::operator*;
        const auto start = std::chrono::high_resolution_clock::now();
        for (auto &particle : particles) {
            const auto molecularMass = particle.getMass();
            const auto force = particle.getForce();
            const auto oldForce = particle.getOldForce();
            const auto changeInVel = (force + oldForce) * (_config.deltaT / (2 * molecularMass));
            particle.addVelocity(changeInVel);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.velocityUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template <typename FloatType>
    void ImplCpp<FloatType>::computeForces(std::vector<Particle<FloatType>> &particles) {
        using namespace ppb::util;
        const size_t size = particles.size();

        const auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < i; ++j) {
                if (i == j) {
                    continue;
                }
                auto &pi = particles[i];
                auto &pj = particles[j];

                const auto sigma = (pi.getSigma() + pj.getSigma()) * 0.5;
                const auto sigmaSquared = sigma * sigma;
                const auto epsilon24 = std::sqrt(pi.getEpsilon() * pj.getEpsilon()) * 24;

                const auto dr = pi.getPosition() - pj.getPosition();
                const auto dr2 = dot(dr, dr);

                const auto invdr2 = 1. / dr2;
                auto lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const auto lj12 = lj6 * lj6;
                const auto lj12m6 = lj12 - lj6;
                const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                const auto f = dr * fac;
                pi.addForce(f);
                pj.subtractForce(f);
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.forceUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::operator-(end, start)).count());
    }

    /* Explicit Instantiation for float and double */
    template class ImplCpp<float>;
    template class ImplCpp<double>;

} // namespace ppb


