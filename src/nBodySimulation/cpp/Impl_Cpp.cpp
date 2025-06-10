#include "NBodySimulation.h"
#include "UtilityContainer.h"
#include <iostream>

template <typename FloatType>
struct ppb::NBodySimulation<FloatType>::impl {

    std::vector<Particle<FloatType>>& particles;
    double &endT;
    double &deltaT;
    std::array<FloatType, 3> &globalForce;

    impl(std::vector<Particle<FloatType>>& particles, double &endT, double &deltaT, std::array<FloatType, 3> &globalForce) :
        particles{particles}, endT{endT}, deltaT{deltaT}, globalForce{globalForce} {}

    void updatePositionsAndResetForce() {
        using namespace ppb::util;
        for (auto &particle : particles) {
            const auto m = particle.getMass();
            auto v = particle.getVelocity();
            auto f = particle.getForce();
            particle.setOldForce(f);
            particle.setForce(globalForce);
            v *= deltaT;
            f *= (deltaT * deltaT / (2 * m));
            const auto displacement = v + f;
            particle.addPosition(displacement);
        }
    }

    void updateVelocities() {
        using namespace ppb::util;
        for (auto &particle : particles) {
            const auto molecularMass = particle.getMass();
            const auto force = particle.getForce();
            const auto oldForce = particle.getOldForce();
            const auto changeInVel = (force + oldForce) * (deltaT / (2 * molecularMass));
            particle.addVelocity(changeInVel);
        }
    }

    void computeForces() {
        using namespace ppb::util;
        const size_t size = particles.size();
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
    }

    void simulate() {
        for (double currentT = 0.0; currentT < endT; currentT += deltaT) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }
    }

};

template<typename FloatType>
void ppb::NBodySimulation<FloatType>::init() {
    _impl = std::make_unique<impl>(_particles, _endT, _deltaT, _globalForce);
}


template <typename FloatType>
typename ppb::NBodySimulation<FloatType>::ParticleContainer ppb::NBodySimulation<FloatType>::operator()() {
    _impl->simulate();
    return _particles;
}


template ppb::NBodySimulation<float>::ParticleContainer ppb::NBodySimulation<float>::operator()();
BENCHMARK(ppb::NBodySimulation<float>::benchmark)
    ->Name("NBody-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

template ppb::NBodySimulation<double>::ParticleContainer ppb::NBodySimulation<double>::operator()();
BENCHMARK(ppb::NBodySimulation<double>::benchmark)
    ->Name("NBody-CStd-Double")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}