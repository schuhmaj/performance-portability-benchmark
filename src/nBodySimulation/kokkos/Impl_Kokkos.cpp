#include "Impl_Kokkos.h"

namespace ppb {

    template <typename FloatType>
    ImplKokkos<FloatType>::ImplKokkos(const ParticleSimulationConfig<FloatType> &config) : _config{config} {

    }

    template<typename FloatType>
    void ImplKokkos<FloatType>::updatePositionsAndResetForce() {
        const auto dt = static_cast<FloatType>(_config.deltaT);

        Kokkos::parallel_for("update_positions", particlesDevice.extent(0), KOKKOS_LAMBDA(const int i) {
            using namespace ppb::util;
            auto& particle = particlesDevice(i);
            const auto m = particle.getMass();
            auto v = particle.getVelocity();
            auto f = particle.getForce();

            particle.setOldForce(f);
            particle.setForce(_config.globalForce);

            v *= dt;
            f *= (dt * dt / (2 * m));
            const auto displacement = v + f;
            particle.addPosition(displacement);
        });
    }

    template<typename FloatType>
    void ImplKokkos<FloatType>::updateVelocities() {
        const auto dt = static_cast<FloatType>(_config.deltaT);

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

    template<typename FloatType>
    void ImplKokkos<FloatType>::computeForces() {
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

    template<typename FloatType>
    std::vector<Particle<FloatType>> ImplKokkos<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<Particle<FloatType>> particlesCopy{particles};
        particlesDevice = ParticleView("particlesDevice", particles.size());
        particlesHost = Kokkos::create_mirror_view(particlesDevice);
        std::copy(particles.begin(), particles.end(), particlesHost.data());

        Kokkos::deep_copy(particlesDevice, particlesHost);

        for (double currentT = 0.0; currentT < _config.endT; currentT += _config.deltaT) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }

        Kokkos::fence();
        Kokkos::deep_copy(particlesHost, particlesDevice);
        for (size_t i = 0; i < particles.size(); ++i) {
            particlesCopy[i] = particlesHost(i);
        }
        return particlesCopy;
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkos<float>;
    template class ImplKokkos<double>;

};