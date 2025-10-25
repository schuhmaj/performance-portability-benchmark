#include "Impl_OpenACC.h"

namespace ppb {

    template <typename FloatType>
    OpenACCParticleSoA<FloatType>::OpenACCParticleSoA(const std::vector<Particle<FloatType>> &ref)
        : _ref{ref}
        , positionsHost(ref.size() * 3, 0.0)
        , velocitiesHost(ref.size() * 3, 0.0)
        , forcesHost(ref.size() * 3, 0.0)
    {
        const size_t n3 = ref.size() * 3;
        // Flatten input Particles into SoA host buffers
        for (size_t i = 0; i < n3; ++i) {
            const size_t particleIndex = i / 3;
            const size_t componentIndex = i % 3;
            positionsHost[i]  = ref[particleIndex].getPosition()[componentIndex];
            velocitiesHost[i] = ref[particleIndex].getVelocity()[componentIndex];
            forcesHost[i]     = ref[particleIndex].getForce()[componentIndex];
        }

        // Allocate and copy device storage using OpenACC data regions.
        // Manage raw device pointers explicitly via acc_malloc/acc_free to avoid mapping host vectors themselves.
        positions  = static_cast<FloatType*>(acc_malloc(n3 * sizeof(FloatType)));
        velocities = static_cast<FloatType*>(acc_malloc(n3 * sizeof(FloatType)));
        forces     = static_cast<FloatType*>(acc_malloc(n3 * sizeof(FloatType)));
        oldForces  = static_cast<FloatType*>(acc_malloc(n3 * sizeof(FloatType)));

        const auto positionsHostPtr = positionsHost.data();
        const auto velocitiesHostPtr = velocitiesHost.data();
        const auto forcesHostPtr = forcesHost.data();

        const auto positionsDevicePtr = positions;
        const auto velocitiesDevicePtr = velocities;
        const auto forcesDevicePtr = forces;
        const auto oldForcesDevicePtr = oldForces;

        #pragma acc enter data copyin(positionsHostPtr[0:n3], velocitiesHostPtr[0:n3], forcesHostPtr[0:n3])
        #pragma acc parallel loop present(positionsHostPtr, velocitiesHostPtr, forcesHostPtr) deviceptr(positionsDevicePtr, velocitiesDevicePtr, forcesDevicePtr, oldForcesDevicePtr)
        for (size_t i = 0; i < n3; ++i) {
            positionsDevicePtr[i]  = positionsHostPtr[i];
            velocitiesDevicePtr[i] = velocitiesHostPtr[i];
            forcesDevicePtr[i]     = forcesHostPtr[i];
            oldForcesDevicePtr[i]  = 0.0;
        }
        #pragma acc exit data delete(positionsHost, velocitiesHost, forcesHost)
    }

    template <typename FloatType>
    OpenACCParticleSoA<FloatType>::~OpenACCParticleSoA() {
        acc_free(positions);
        acc_free(velocities);
        acc_free(forces);
        acc_free(oldForces);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> OpenACCParticleSoA<FloatType>::toParticles() {
        const size_t n = _ref.size();
        const size_t n3 = n * 3;

        const auto positionsHostPtr = positionsHost.data();
        const auto velocitiesHostPtr = velocitiesHost.data();
        const auto forcesHostPtr = forcesHost.data();

        const auto positionsDevicePtr = positions;
        const auto velocitiesDevicePtr = velocities;
        const auto forcesDevicePtr = forces;

        // Copy device data back into host mirrors
        #pragma acc parallel loop deviceptr(positionsDevicePtr, velocitiesDevicePtr, forcesDevicePtr) copyout(positionsHostPtr[0:n3], velocitiesHostPtr[0:n3], forcesHostPtr[0:n3])
        for (size_t i = 0; i < n3; ++i) {
            positionsHostPtr[i]  = positionsDevicePtr[i];
            velocitiesHostPtr[i] = velocitiesDevicePtr[i];
            forcesHostPtr[i]     = forcesDevicePtr[i];
        }

        std::vector<Particle<FloatType>> particles{_ref};
        for (size_t i = 0; i < n; ++i) {
            std::array<FloatType, 3> pos{};
            std::array<FloatType, 3> vel{};
            std::array<FloatType, 3> force{};
            for (size_t j = 0; j < 3; ++j) {
                pos[j]   = positionsHost[i * 3 + j];
                vel[j]   = velocitiesHost[i * 3 + j];
                force[j] = forcesHost[i * 3 + j];
            }
            particles[i].setPosition(pos);
            particles[i].setVelocity(vel);
            particles[i].setForce(force);
        }
        return particles;
    }


    template <typename FloatType>
    ImplOpenACC<FloatType>::ImplOpenACC(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplOpenACC<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
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
    void ImplOpenACC<FloatType>::updatePositionsAndResetForce() {
        const size_t size = _config.size;
        const FloatType dt = _config.deltaT;
        const std::array<float_type, 3> &globalForce = _config.globalForce;
        auto *forces = _particles->forces;
        auto *oldForces = _particles->oldForces;
        const auto *velocities = _particles->velocities;
        auto *positions = _particles->positions;

        constexpr FloatType m = 1.0;
        const FloatType tt2m = (dt * dt / (2 * m));

        const auto start = std::chrono::high_resolution_clock::now();
        #pragma acc parallel loop collapse(2) deviceptr(positions, velocities, forces, oldForces)
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                const size_t index = i * 3 + j;
                const auto v = velocities[index];
                const auto f = forces[index];
                oldForces[index] = f;
                forces[index] = globalForce[j];
                positions[index] += (v * dt + f * tt2m);
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.positionUpdateForceResetTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template <typename FloatType>
    void ImplOpenACC<FloatType>::updateVelocities() {
        const size_t size = _config.size;
        const FloatType dt = _config.deltaT;
        const auto *forces = _particles->forces;
        const auto *oldForces = _particles->oldForces;
        auto *velocities = _particles->velocities;

        constexpr FloatType m = 1.0;
        const FloatType t2m = (dt / (2.0 * m));

        const auto start = std::chrono::high_resolution_clock::now();
        #pragma acc parallel loop collapse(2) deviceptr(velocities, forces, oldForces)
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                const size_t index = i * 3 + j;
                velocities[index] += ((forces[index] + oldForces[index]) * t2m);
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.velocityUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template <typename FloatType>
    void ImplOpenACC<FloatType>::computeForces() {
        const size_t size = _config.size;
        auto *forces = _particles->forces;
        const auto *positions = _particles->positions;

        const auto start = std::chrono::high_resolution_clock::now();
        #pragma acc parallel loop deviceptr(forces, positions)
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (i == j) continue;

                constexpr FloatType sigma = (1.0 + 1.0) * 0.5;
                constexpr FloatType sigmaSquared = sigma * sigma;
                const FloatType epsilon24 = std::sqrt(1.0 * 1.0) * 24;

                FloatType dr0 = positions[i * 3 + 0] - positions[j * 3 + 0];
                FloatType dr1 = positions[i * 3 + 1] - positions[j * 3 + 1];
                FloatType dr2 = positions[i * 3 + 2] - positions[j * 3 + 2];
                const FloatType r2 = dr0 * dr0 + dr1 * dr1 + dr2 * dr2;

                const FloatType invr2 = 1.0 / r2;
                FloatType lj6 = sigmaSquared * invr2;
                lj6 = lj6 * lj6 * lj6;
                const FloatType lj12 = lj6 * lj6;
                const FloatType lj12m6 = lj12 - lj6;
                const FloatType fac = epsilon24 * (lj12 + lj12m6) * invr2;

                forces[i * 3 + 0] += dr0 * fac;
                forces[i * 3 + 1] += dr1 * fac;
                forces[i * 3 + 2] += dr2 * fac;
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        _timings.forceUpdateTime += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    template class ImplOpenACC<float>;
    template class OpenACCParticleSoA<float>;

} // namespace ppb