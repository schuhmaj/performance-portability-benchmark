#include "Impl_Alpaka.h"
#include <chrono>

template <typename FloatType>
ppb::AlpakaParticleSoA<FloatType>::AlpakaParticleSoA(const std::vector<Particle<FloatType>> &ref, Host &host, Device &device, Queue &queue)
    : _ref{ref}
    , extent{ref.size() * 3}
    , positions{alpaka::allocBuf<FloatType, Idx>(device, extent)}
    , velocities{alpaka::allocBuf<FloatType, Idx>(device, extent)}
    , forces{alpaka::allocBuf<FloatType, Idx>(device, extent)}
    , oldForces{alpaka::allocBuf<FloatType, Idx>(device, extent)}
    , positionsHost{alpaka::allocBuf<FloatType, Idx>(host, extent)}
    , velocitiesHost{alpaka::allocBuf<FloatType, Idx>(host, extent)}
    , forcesHost{alpaka::allocBuf<FloatType, Idx>(host, extent)}
{
    auto posPtr = alpaka::getPtrNative(positionsHost);
    auto velPtr = alpaka::getPtrNative(velocitiesHost);
    auto frcPtr = alpaka::getPtrNative(forcesHost);
    for (Idx i = 0; i < _ref.size(); ++i) {
        for (Idx j = 0; j < 3; ++j) {
            posPtr[i * 3 + j] = ref[i].getPosition()[j];
            velPtr[i * 3 + j] = ref[i].getVelocity()[j];
            frcPtr[i * 3 + j] = ref[i].getForce()[j];
        }
    }
    alpaka::memcpy(queue, positions, positionsHost, extent);
    alpaka::memcpy(queue, velocities, velocitiesHost, extent);
    alpaka::memcpy(queue, forces, forcesHost, extent);
    alpaka::fill(queue, oldForces, 0.0f, extent);
    alpaka::wait(queue);
}

template <typename FloatType>
std::vector<ppb::Particle<FloatType>> ppb::AlpakaParticleSoA<FloatType>::toParticles(Queue &queue) {
    std::vector<Particle<FloatType>> out{_ref};
    alpaka::memcpy(queue, positionsHost, positions, extent);
    alpaka::memcpy(queue, velocitiesHost, velocities, extent);
    alpaka::memcpy(queue, forcesHost, forces, extent);
    alpaka::wait(queue);

    auto posPtr = alpaka::getPtrNative(positionsHost);
    auto velPtr = alpaka::getPtrNative(velocitiesHost);
    auto frcPtr = alpaka::getPtrNative(forcesHost);
    for (Idx i = 0; i < _ref.size(); ++i) {
        std::array<FloatType, 3> p{}, v{}, f{}, of{};
        for (Idx j = 0; j < 3; ++j) {
            p[j] = posPtr[i * 3 + j];
            v[j] = velPtr[i * 3 + j];
            f[j] = frcPtr[i * 3 + j];
        }
        out[i].setPosition(p);
        out[i].setVelocity(v);
        out[i].setForce(f);
    }
    return out;
}

template<typename FloatType>
ppb::ImplAlpaka<FloatType>::ImplAlpaka(const ParticleSimulationConfig<FloatType> &config)
    : _config{config}
    , _timings{}
    , host(alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0))
    , device(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0))
    , queue(device) {}


template <typename FloatType>
std::pair<std::vector<ppb::Particle<FloatType>>, ppb::ParticleSimulationTimings> ppb::ImplAlpaka<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
    _timings.reset();
    _particles.emplace(particles, host, device, queue);

    for (int i = 0; i < _config.numberTimeSteps; ++i) {
        updatePositionsAndResetForce();
        computeForces();
        updateVelocities();
    }
    return std::make_pair(_particles->toParticles(queue), _timings);
}

template <typename FloatType>
void ppb::ImplAlpaka<FloatType>::updatePositionsAndResetForce() {
    const size_t n = _config.size;
    const std::array<float_type, 3> &globalForce = _config.globalForce;
    const auto &dt = static_cast<float_type>(_config.deltaT);

    const auto kernel = [=] ALPAKA_FN_ACC (const Acc& acc, float_type* positions, float_type* velocities, float_type* forces, float_type *oldForces, Idx numParticles) {
        const auto i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if (i >= numParticles) {
            return;
        }
        constexpr float_type m = 1.0;
        const float_type tt2m = (dt * dt / (2.0 * m));

        float_type force[3];
        float_type velocityPart[3];
        float_type forcePart[3];
        for (unsigned int d = 0; d < 3; ++d) {
            const unsigned int index = i * 3 + d;
            force[d] = forces[index];
            oldForces[index] = force[d];
            forces[index] = globalForce[d];

            velocityPart[d] = velocities[index] * dt;
            forcePart[d] = force[d] * tt2m;

            positions[index] += (velocityPart[d] + forcePart[d]);
        }
    };

    auto posPtr = alpaka::getPtrNative(_particles->positions);
    auto velPtr = alpaka::getPtrNative(_particles->velocities);
    auto frcPtr = alpaka::getPtrNative(_particles->forces);
    auto oldPtr = alpaka::getPtrNative(_particles->oldForces);

    const auto workDiv = alpaka::getValidWorkDiv(
        alpaka::KernelCfg<Acc>{n, 1},
        device, kernel, posPtr, velPtr, frcPtr, oldPtr, n);
    auto task = alpaka::createTaskKernel<Acc>(workDiv, kernel, posPtr, velPtr, frcPtr, oldPtr, n);

    const auto start = std::chrono::high_resolution_clock::now();
    alpaka::enqueue(queue, task);
    alpaka::wait(queue);
    const auto end = std::chrono::high_resolution_clock::now();
    _timings.positionUpdateForceResetTime += static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

template <typename FloatType>
void ppb::ImplAlpaka<FloatType>::updateVelocities() {
    const size_t n = _config.size;
    const auto &dt = static_cast<float_type>(_config.deltaT);

    const auto kernel = [=] ALPAKA_FN_ACC (const Acc& acc, float_type* vel, float_type* force, float_type *oldForce, Idx numParticles) {
        const auto i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if (i >= numParticles) {
            return;
        }
        constexpr float_type m = 1.0;
        const float_type t2m = (dt / (2.0 * m));

        for (int d = 0; d < 3; ++d) {
            const unsigned int index = i * 3 + d;
            vel[index] += ((force[index] + oldForce[index]) * t2m);
        }
    };

    auto velPtr = alpaka::getPtrNative(_particles->velocities);
    auto frcPtr = alpaka::getPtrNative(_particles->forces);
    auto oldPtr = alpaka::getPtrNative(_particles->oldForces);

    const auto workDiv = alpaka::getValidWorkDiv(
        alpaka::KernelCfg<Acc>{n, 1},
        device, kernel, velPtr, frcPtr, oldPtr, n);
    auto task = alpaka::createTaskKernel<Acc>(workDiv, kernel, velPtr, frcPtr, oldPtr, n);

    const auto start = std::chrono::high_resolution_clock::now();
    alpaka::enqueue(queue, task);
    alpaka::wait(queue);
    const auto end = std::chrono::high_resolution_clock::now();
    _timings.velocityUpdateTime += static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

template <typename FloatType>
void ppb::ImplAlpaka<FloatType>::computeForces() {
    const size_t n = _config.size;
    const auto &dt = static_cast<float_type>(_config.deltaT);

    const auto kernel = [=] ALPAKA_FN_ACC (const Acc& acc, float_type* pos, float_type* force, Idx numParticles) {
        const auto i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if (i >= numParticles) {
            return;
        }

        constexpr float_type sigmaSrc = 1.0;
        constexpr float_type epsilonSrc = 1.0;
        constexpr float_type sigma = (sigmaSrc + sigmaSrc) * 0.5;
        constexpr float_type sigmaSquared = sigma * sigma;
        const float_type epsilon24 = alpaka::math::sqrt(acc, epsilonSrc * epsilonSrc) * 24.0f;

        float_type accumulator[3] = {0.0f, 0.0f, 0.0f};
        for (Idx j = 0; j < numParticles; ++j) {
            if (i == j) {
                continue;
            }
            float_type dr[3];
            float_type dr2 = 0.0f;
            for (int d = 0; d < 3; ++d) {
                const unsigned int indexI = i * 3 + d;
                const unsigned int indexJ = j * 3 + d;
                dr[d] = pos[indexI] - pos[indexJ];
                dr2 += dr[d] * dr[d];
            }

            const float_type invdr2 = 1.0f / dr2;
            float_type lj6 = sigmaSquared * invdr2;
            lj6 = lj6 * lj6 * lj6;
            const float_type lj12 = lj6 * lj6;
            const float_type lj12m6 = lj12 - lj6;
            const float_type fac = epsilon24 * (lj12 + lj12m6) * invdr2;

            for (int d = 0; d < 3; ++d) {
                accumulator[d] += (dr[d] * fac);
            }
        }

        for (int d = 0; d < 3; ++d) {
            force[i * 3 + d] += accumulator[d];
        }
    };

    auto posPtr = alpaka::getPtrNative(_particles->positions);
    auto frcPtr = alpaka::getPtrNative(_particles->forces);

    const auto workDiv = alpaka::getValidWorkDiv(
        alpaka::KernelCfg<Acc>{n, 1},
        device, kernel, posPtr, frcPtr, n);
    auto task = alpaka::createTaskKernel<Acc>(workDiv, kernel, posPtr, frcPtr, n);

    const auto start = std::chrono::high_resolution_clock::now();
    alpaka::enqueue(queue, task);
    alpaka::wait(queue);
    const auto end = std::chrono::high_resolution_clock::now();
    _timings.forceUpdateTime += static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

template class ppb::ImplAlpaka<float>;
template class ppb::AlpakaParticleSoA<float>;

template class ppb::ImplAlpaka<double>;
template class ppb::AlpakaParticleSoA<double>;