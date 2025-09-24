#include "Impl_AdaptiveCpp.h"
#include <sycl/sycl.hpp>

namespace ppb {

    sycl::queue queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

    template <typename FloatType>
    ImplAdaptiveCpp<FloatType>::ImplAdaptiveCpp(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::vector<Particle<FloatType>> ImplAdaptiveCpp<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<Particle<FloatType>> particlesCopy{particles};
        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce(particlesCopy);
            computeForces(particlesCopy);
            updateVelocities(particlesCopy);
        }
        return particlesCopy;
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updatePositionsAndResetForce(std::vector<Particle<FloatType>> &particles) {
        using namespace ppb::util;
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
    }
    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updateVelocities(std::vector<Particle<FloatType>> &particles) {
        using namespace ppb::util;
        for (auto &particle : particles) {
            const auto molecularMass = particle.getMass();
            const auto force = particle.getForce();
            const auto oldForce = particle.getOldForce();
            const auto changeInVel = (force + oldForce) * (_config.deltaT / (2 * molecularMass));
            particle.addVelocity(changeInVel);
        }
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::computeForces(std::vector<Particle<FloatType>> &particles) {
        using namespace ppb::util;
        const size_t size = particles.size();

        Particle<FloatType> *particlesUSM = sycl::malloc_shared<Particle<FloatType>>(size, queue);
        if (!particlesUSM) {
            throw std::runtime_error("USM allocation failed");
        }

        std::memcpy(particlesUSM, particles.data(), size * sizeof(Particle<FloatType>));

        queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            auto &pi = particlesUSM[i];

            pi.setForce({FloatType(0), FloatType(0), FloatType(0)});

            for (size_t j = 0; j < size; ++j) {
                if (i == j) continue;
                const auto &pj = particlesUSM[j];

                const auto sigma = FloatType(1); //(pi.getSigma() + pj.getSigma()) * FloatType(0.5);
                const auto sigmaSquared = sigma * sigma;
                const auto epsilon24 = FloatType(120); //sycl::sqrt(pi.getEpsilon() * pj.getEpsilon()) * FloatType(24);

                const auto dr = pi.getPosition() - pj.getPosition();
                const auto dr2 = dot(dr, dr);

                const auto invdr2 = FloatType(1) / dr2;
                auto lj6 = sigmaSquared * invdr2;
                lj6 = lj6 * lj6 * lj6;
                const auto lj12 = lj6 * lj6;
                const auto lj12m6 = lj12 - lj6;
                const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                const auto f = dr * fac;
                pi.addForce(f);
            }
        });

        queue.wait();

        std::memcpy(particles.data(), particlesUSM, size * sizeof(Particle<FloatType>));

        sycl::free(particlesUSM, queue);

        /*
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (i == j) {
                    continue;
                }
                auto &pi = particles[i];
                auto &pj = particles[j];

                const auto sigma = pi.getSigma() * pj.getSigma() * 0.5;
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
            }
        }
        */
    }

    /* Explicit Instantiation for float and double */
    template class ImplAdaptiveCpp<float>;
    template class ImplAdaptiveCpp<double>;

} // namespace ppb


