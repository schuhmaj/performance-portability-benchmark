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
        , oldForcesHost{Kokkos::create_mirror_view(oldForces)}
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
        Kokkos::deep_copy(oldForcesHost, oldForces);
        for (size_t i = 0; i < particles.size(); ++i) {
            std::array<FloatType, 3> pos{};
            std::array<FloatType, 3> vel{};
            std::array<FloatType, 3> force{};
            std::array<FloatType, 3> oldForce{};
            for (size_t j = 0; j < 3; ++j) {
                pos[j] = positionsHost(i, j);
                vel[j] = velocitiesHost(i, j);
                force[j] = forcesHost(i, j);
                oldForce[j] = oldForcesHost(i, j);
            }
            particles[i].setPosition(pos);
            particles[i].setVelocity(vel);
            particles[i].setForce(force);
            particles[i].setOldForce(oldForce);
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
        const size_t size = _particles->size();
        constexpr size_t dim = 3;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        const auto &globalForce = _config.globalForce;
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;
        auto &position = _particles->positions;

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {size, dim});
        const Kokkos::Timer timer;
        Kokkos::parallel_for("update_positions", policy, KOKKOS_LAMBDA(const int i, const int j) {
            using namespace ppb::util;
            const auto m = 1.0;
            auto v = velocity(i, j);
            auto f = force(i, j);

            oldForce(i, j) = f;
            force(i, j) = globalForce[j];

            v *= dt;
            f *= (dt * dt / (2 * m));
            const auto displacement = v + f;
            position(i, j) = position(i, j) + displacement;
        });
        _timings.positionUpdateForceResetTime += (timer.seconds() * 1e9);
    }

    template<typename FloatType>
    void ImplKokkos<FloatType>::updateVelocities() {
        const size_t size = _particles->size();
        constexpr size_t dim = 3;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {size, dim});
        const Kokkos::Timer timer;
        Kokkos::parallel_for("update_velocities", policy, KOKKOS_LAMBDA(const int i, const int j) {
            using namespace ppb::util;
            constexpr auto m = 1.0;
            const auto changeInVel = (force(i, j) + oldForce(i, j)) * (dt / (2 * m));
            velocity(i, j) = velocity(i, j) + changeInVel;
        });
        Kokkos::fence();
        _timings.velocityUpdateTime += (timer.seconds() * 1e9);
    }

    template<typename FloatType>
    void ImplKokkos<FloatType>::computeForces() {
        const size_t size = _particles->size();
        auto &force = _particles->forces;
        auto &position = _particles->positions;

        using TeamPolicy = Kokkos::TeamPolicy<>;
        using MemberType = TeamPolicy::member_type;

        TeamPolicy policy(size, Kokkos::AUTO);

        const Kokkos::Timer timer;
        Kokkos::parallel_for("compute_forces_team", policy, KOKKOS_LAMBDA(const MemberType& team) {
            const int i = team.league_rank();

            constexpr auto sigmaSrc = 1.0;
            constexpr auto epsilonSrc = 1.0;

            constexpr auto sigma = (sigmaSrc + sigmaSrc) * 0.5;
            constexpr auto sigmaSquared = sigma * sigma;
            const auto epsilon24 = Kokkos::sqrt(epsilonSrc * epsilonSrc) * 24.0;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, 0, i), [&](const int j) {
                    if (j == i) {
                        return;
                    }
                    std::array<FloatType, 3> dr{};
                    FloatType dr2 = 0;
                    for (int k = 0; k < 3; ++k) {
                        dr[k] = position(i, k) - position(j, k);
                        dr2 += dr[k] * dr[k];
                    }
                    const auto invdr2 = 1.0 / dr2;
                    auto lj6 = sigmaSquared * invdr2;
                    lj6 = lj6 * lj6 * lj6;
                    const auto lj12 = lj6 * lj6;
                    const auto lj12m6 = lj12 - lj6;
                    const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                    for (int k = 0; k < 3; ++k) {
                        const auto f = dr[k] * fac;
                        Kokkos::atomic_add(&force(i, k), f);
                        Kokkos::atomic_sub(&force(j, k), f);
                    }
                }
            );
        });
        Kokkos::fence();
        _timings.forceUpdateTime += (timer.seconds() * 1e9);
    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplKokkos<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }

        Kokkos::fence();
        return std::make_pair(_particles->toParticles(), _timings);
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkos<float>;
    template class ImplKokkos<double>;

};