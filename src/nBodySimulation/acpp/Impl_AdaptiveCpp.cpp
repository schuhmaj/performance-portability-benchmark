#include "Impl_AdaptiveCpp.h"

namespace ppb {

    template <typename FloatType>
    AdaptiveCppParticleSoA<FloatType>::AdaptiveCppParticleSoA(const std::vector<Particle<FloatType>> &particles, sycl::queue &queue)
        : _ref{particles}
        , _queue{queue}
        , positions{sycl::malloc_shared<FloatType>(particles.size() * 3, queue)}
        , velocities{sycl::malloc_shared<FloatType>(particles.size() * 3, queue)}
        , forces{sycl::malloc_shared<FloatType>(particles.size() * 3, queue)}
        , oldForces{sycl::malloc_device<FloatType>(particles.size() * 3, queue)}
    {
        for (size_t i = 0; i < particles.size() * 3; ++i) {
            const size_t particleIndex = i / 3;
            const size_t componentIndex = i % 3;
            positions[i] = particles[particleIndex].getPosition()[componentIndex];
            velocities[i] = particles[particleIndex].getVelocity()[componentIndex];
            forces[i] = particles[particleIndex].getForce()[componentIndex];
        }
        queue.fill(oldForces, 0.0f, particles.size() * 3);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> AdaptiveCppParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        for (size_t i = 0; i < particles.size(); ++i) {
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

    template<typename FloatType>
    ImplAdaptiveCpp<FloatType>::ImplAdaptiveCpp(const ParticleSimulationConfig<FloatType> &config) : _config{config}, _queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}} {

    }

    template<typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplAdaptiveCpp<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles, _queue);

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }

        return std::make_pair(_particles->toParticles(), _timings);
    }

    template<typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        constexpr size_t dim = 3;
        const auto dt = static_cast<FloatType>(_config.deltaT);
        const auto &globalForce = _config.globalForce;
        auto &forces = _particles->forces;
        auto &oldForces = _particles->oldForces;
        auto &velocities = _particles->velocities;
        auto &positions = _particles->positions;


        auto event = _queue.submit([&](sycl::handler &h) {
        const sycl::range<1> local{32};
        const sycl::range<1> global{util::roundUp(size, local[0])};
            h.parallel_for(sycl::nd_range<1>{global, local}, [=](const sycl::nd_item<1> &it) {
                const size_t i = it.get_global_id(0);
                if (i >= size) return;

                constexpr FloatType m = 1.0;
                const FloatType tt2m = (dt * dt / (2 * m));
                for (int d = 0; d < dim; ++d) {
                    const FloatType force = forces[i * 3 + d];
                    oldForces[i * 3 + d] = force;
                    forces[i * 3 + d] = globalForce[d];


                    FloatType v = velocities[i * 3 + d] * dt;
                    FloatType f = force * (dt * dt / (2 * m));
                    FloatType displacement = v + f;
                    positions[i * 3 + d] += displacement;
                }
            });
        });
        event.wait_and_throw();
        auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const double elapsed_nanoseconds = end - start;
        _timings.positionUpdateForceResetTime += elapsed_nanoseconds;
    }

    template<typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        constexpr size_t dim = 3;
        const FloatType dt = static_cast<FloatType>(_config.deltaT);
        auto &force = _particles->forces;
        auto &oldForce = _particles->oldForces;
        auto &velocity = _particles->velocities;

        auto event = _queue.submit([&](sycl::handler &h) {
            const sycl::range<1> local{32};
            const sycl::range<1> global{util::roundUp(size, local[0])};
            h.parallel_for(sycl::nd_range<1>{global, local}, [=](const sycl::nd_item<1> &it) {
                const size_t i = it.get_global_id(0);
                if (i >= size) return;

                constexpr FloatType m = 1.0;
                const FloatType t2m = dt / (2 * m);
                for (int d = 0; d < dim; ++d) {
                    const size_t idx = i * 3 + d;
                    const FloatType dv = (force[idx] + oldForce[idx]) * t2m;
                    velocity[idx] += dv;
                }
            });
        });
        event.wait_and_throw();
        auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const double elapsed_nanoseconds = end - start;
        _timings.velocityUpdateTime += elapsed_nanoseconds;
    }

    template<typename FloatType>
    void ImplAdaptiveCpp<FloatType>::computeForces() {
        const size_t size = _config.size;
        auto &forces = _particles->forces;
        auto &positions = _particles->positions;


        auto event = _queue.submit([&](sycl::handler &h) {
            const sycl::range<1> local{32};
            const sycl::range<1> global{util::roundUp(size, local[0])};
            h.parallel_for(sycl::nd_range<1>{global, local}, [=](const sycl::nd_item<1>& it) {
                const size_t i = it.get_global_id(0);
                if (i >= size) return;

                constexpr FloatType sigma = static_cast<FloatType>(1.0);
                constexpr FloatType epsilon24 = static_cast<FloatType>(24.0);

                FloatType accx = 0, accy = 0, accz = 0;
                const FloatType pix = positions[i * 3 + 0];
                const FloatType piy = positions[i * 3 + 1];
                const FloatType piz = positions[i * 3 + 2];

                for (size_t j = 0; j < size; ++j) {
                    if (i == j) continue;
                    const FloatType drx = pix - positions[j * 3 + 0];
                    const FloatType dry = piy - positions[j * 3 + 1];
                    const FloatType drz = piz - positions[j * 3 + 2];
                    const FloatType dr2 = drx*drx + dry*dry + drz*drz;
                    const FloatType invdr2 = static_cast<FloatType>(1.0) / dr2;
                    FloatType lj6 = (sigma * sigma) * invdr2;
                    lj6 = lj6 * lj6 * lj6;
                    const FloatType lj12 = lj6 * lj6;
                    const FloatType lj12m6 = lj12 - lj6;
                    const FloatType fac = epsilon24 * (lj12 + lj12m6) * invdr2;
                    accx += drx * fac;
                    accy += dry * fac;
                    accz += drz * fac;
                }
                forces[i * 3 + 0] += accx;
                forces[i * 3 + 1] += accy;
                forces[i * 3 + 2] += accz;
            });
        });
        event.wait_and_throw();
        auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const double elapsed_nanoseconds = end - start;
        _timings.forceUpdateTime += elapsed_nanoseconds;
    }


    template class AdaptiveCppParticleSoA<float>;
    template class ImplAdaptiveCpp<float>;




};