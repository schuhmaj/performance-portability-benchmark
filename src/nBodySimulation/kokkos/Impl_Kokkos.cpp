#include <Kokkos_Core.hpp>
#define FUNCTION_PREFIX KOKKOS_FUNCTION

#include <Kokkos_Core.hpp>
#include <iostream>
#include "NBodySimulation.h"
#include "UtilityContainer.h"

#include "CSVFileHandler.h"

template <typename FloatType>
struct ppb::NBodySimulation<FloatType>::impl {

    std::vector<Particle<FloatType>>& particles;
    double &endT;
    double &deltaT;
    std::array<FloatType, 3> globalForce;

    using ParticleView = Kokkos::View<Particle<FloatType>*>;
    ParticleView particlesDevice;
    typename ParticleView::HostMirror particlesHost;

    Kokkos::View<FloatType[3]> globalForceDevice;
    typename decltype(globalForceDevice)::HostMirror globalForceHost;


    impl(std::vector<Particle<FloatType>>& particles, double &endT, double &deltaT, std::array<FloatType, 3> &globalForce) :
        particles{particles}, endT{endT}, deltaT{deltaT}, globalForce{globalForce} {

        particlesDevice = ParticleView("particlesDevice", particles.size());
        particlesHost = Kokkos::create_mirror_view(particlesDevice);
        std::copy(particles.begin(), particles.end(), particlesHost.data());

        globalForceDevice = Kokkos::View<FloatType[3]>("globalForce");
        globalForceHost = Kokkos::create_mirror_view(globalForceDevice);
        std::copy(globalForce.begin(), globalForce.end(), globalForceHost.data());

        Kokkos::deep_copy(particlesDevice, particlesHost);
        Kokkos::deep_copy(globalForceDevice, globalForceHost);
    }

    void updatePositionsAndResetForce() {
        const auto dt = static_cast<FloatType>(deltaT);

        Kokkos::parallel_for("update_positions", particlesDevice.extent(0), KOKKOS_LAMBDA(const int i) {
            using namespace ppb::util;
            auto& particle = particlesDevice(i);
            const auto m = particle.getMass();
            auto v = particle.getVelocity();
            auto f = particle.getForce();

            particle.setOldForce(f);
            particle.setForce(globalForce);

            v *= dt;
            f *= (dt * dt / (2 * m));
            const auto displacement = v + f;
            particle.addPosition(displacement);
        });
    }

    void updateVelocities() {
        const auto dt = static_cast<FloatType>(deltaT);

        Kokkos::parallel_for("update_velocities", particlesDevice.extent(0), KOKKOS_LAMBDA(const int i) {
            using namespace ppb::util;
            auto& particle = particlesDevice(i);
            const auto molecularMass = particle.getMass();
            const auto force = particle.getForce();
            const auto oldForce = particle.getOldForce();
            const auto changeInVel = (force + oldForce) * (dt / (2 * molecularMass));
            particle.addVelocity(changeInVel);
        });
    }

    void computeForces() {
        Kokkos::parallel_for("compute_forces", particlesDevice.extent(0), KOKKOS_LAMBDA(const int i) {
            using namespace ppb::util;
            auto& pi = particlesDevice(i);
            const auto pi_pos = pi.getPosition();

            for (int j = 0; j < particlesDevice.extent(0); ++j) {
                if (i == j) continue;

                const auto& pj = particlesDevice(j);

                const auto sigma = pi.getSigma() * pj.getSigma() * 0.5;
                const auto sigmaSquared = sigma * sigma;
                const auto epsilon24 = Kokkos::sqrt(pi.getEpsilon() * pj.getEpsilon()) * 24.0;

                const auto dr = pi_pos - pj.getPosition();
                const auto dr2 = ppb::util::dot(dr, dr);

                if (dr2 > 1e-12) {
                    const auto invdr2 = 1.0 / dr2;
                    auto lj6 = sigmaSquared * invdr2;
                    lj6 = lj6 * lj6 * lj6;
                    const auto lj12 = lj6 * lj6;
                    const auto lj12m6 = lj12 - lj6;
                    const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                    const auto f = dr * fac;
                    pi.addForce(f);
                }
            }
        });
    }

    void simulate() {
        for (double currentT = 0.0; currentT < endT; currentT += deltaT) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }
        Kokkos::fence();
        Kokkos::deep_copy(particlesHost, particlesDevice);
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i] = particlesHost(i);
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
    ->Name("NBody-Kokkos-Float")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

template ppb::NBodySimulation<double>::ParticleContainer ppb::NBodySimulation<double>::operator()();
BENCHMARK(ppb::NBodySimulation<double>::benchmark)
    ->Name("NBody-Kokkos-Double")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}