#include "Impl_Kokkos.h"

namespace ppb {

    template <typename FloatType>
    KokkosParticleSoA<FloatType>::KokkosParticleSoA(const std::vector<Particle<FloatType>> &particles)
        : positions{"positionsDevice", particles.size()}
        , positionsHost{Kokkos::create_mirror_view(positions)}
        , velocities{"velocitiesDevice", particles.size()}
        , velocitiesHost{Kokkos::create_mirror_view(velocities)}
        , forces{"forcesDevice", particles.size()}
        , forcesHost{Kokkos::create_mirror_view(forces)}
        , oldForces{"oldForcesDevice", particles.size()}
        , types{"typesDevice", particles.size()}
        , _ref{particles}
    {
        for (size_t i = 0; i < particles.size(); ++i) {
            for (size_t j = 0; j < 3; ++j) {
                positionsHost(i, j) = particles[i].getPosition()[j];
                velocitiesHost(i, j) = particles[i].getVelocity()[j];
                forcesHost(i, j) = particles[i].getForce()[j];
            }
        }
        Kokkos::deep_copy(positions, positionsHost);
        Kokkos::deep_copy(velocities, velocitiesHost);
        Kokkos::deep_copy(forces, forcesHost);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> KokkosParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        Kokkos::deep_copy(positionsHost, positions);
        Kokkos::deep_copy(velocitiesHost, velocities);
        Kokkos::deep_copy(forcesHost, forces);
        for (size_t i = 0; i < particles.size(); ++i) {
            std::array<FloatType, 3> pos{};
            std::array<FloatType, 3> vel{};
            std::array<FloatType, 3> force{};
            for (size_t j = 0; j < 3; ++j) {
                pos[j] = positionsHost(i, j);
                vel[j] = velocitiesHost(i, j);
                force[j] = forcesHost(i, j);
            }
            particles[i].setPosition(pos);
            particles[i].setVelocity(vel);
            particles[i].setForce(force);
        }
        return particles;
    }

    template <typename FloatType>
    size_t KokkosParticleSoA<FloatType>::size() const {
        return _ref.size();
    }


    template class KokkosParticleSoA<float>;
    template class KokkosParticleSoA<double>;



    template<typename FloatType>
    ImplKokkos<FloatType>::ImplKokkos(const ParticleSimulationConfig<FloatType> &config) : _config{config} {

    }

    template<typename FloatType>
    void ImplKokkos<FloatType>::updatePositionsAndResetForce() {
        const auto dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;

        Kokkos::parallel_for("update_positions", _particles->size(), KOKKOS_LAMBDA(const int i) {
            using namespace ppb::util;
            const auto m = 1.0;
            for (int j = 0; j < 3; ++j) {
                auto v = velocity(i, j);
                auto f = force(i, j);

                oldForce(i, j) = f;
                force(i, j) = _config.globalForce[j];

                v *= dt;
                f *= (dt * dt / (2 * m));
                const auto displacement = v + f;
                position(i, j) = position(i, j) + displacement;
            }

        });
    }

    template<typename FloatType>
    void ImplKokkos<FloatType>::updateVelocities() {
        const auto dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        Kokkos::parallel_for("update_velocities", _particles->size(), KOKKOS_LAMBDA(const int i) {
            using namespace ppb::util;
            for (int j = 0; j < 3; ++j) {
                constexpr auto m = 1.0;
                const auto changeInVel = (force(i, j) + oldForce(i, j)) * (dt / (2 * m));
                velocity(i, j) = velocity(i, j) + changeInVel;
            }
        });
    }

    template<typename FloatType>
    void ImplKokkos<FloatType>::computeForces() {
        const size_t size = _particles->size();
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;

        Kokkos::parallel_for("compute_forces", size, KOKKOS_LAMBDA(const int i) {
            using namespace ppb::util;

            for (int j = 0; j < size; ++j) {
                if (i == j) continue;

                constexpr auto sigmaSrc = 1.0;
                constexpr auto epsilonSrc = 5.0;

                constexpr auto sigma = sigmaSrc * sigmaSrc * 0.5;
                const auto sigmaSquared = sigma * sigma;
                const auto epsilon24 = Kokkos::sqrt(epsilonSrc * epsilonSrc) * 24.0;

                std::array<FloatType, 3> dr{};
                for (int k = 0; k < 3; ++k) {
                    dr[k] = position(i, k) - position(j, k);
                }
                const auto dr2 = ppb::util::dot(dr, dr);

                for (int k = 0; k < 3; ++k) {
                    const auto invdr2 = 1.0 / dr2;
                    auto lj6 = sigmaSquared * invdr2;
                    lj6 = lj6 * lj6 * lj6;
                    const auto lj12 = lj6 * lj6;
                    const auto lj12m6 = lj12 - lj6;
                    const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                    const auto f = dr[k] * fac;
                    force(i, k) = force(i, k) + f;
                }
            }
        });
    }

    template<typename FloatType>
    std::vector<Particle<FloatType>> ImplKokkos<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _particles.emplace(particles);

        for (double currentT = 0.0; currentT < _config.endT; currentT += _config.deltaT) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }

        Kokkos::fence();
        return _particles->toParticles();
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkos<float>;
    template class ImplKokkos<double>;

};