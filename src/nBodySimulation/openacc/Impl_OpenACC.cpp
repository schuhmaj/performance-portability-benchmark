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

        // Allocate raw device storage and transfer the flattened host buffers with the
        // OpenACC runtime copy API. Using acc_memcpy_to_device (rather than mapping the
        // host vectors via enter/exit data) keeps the device pointers fully self-managed
        // and avoids leaving stale present-table entries that would suppress later copies.
        const size_t numberOfBytes = n3 * sizeof(FloatType);
        positions  = static_cast<FloatType*>(acc_malloc(numberOfBytes));
        velocities = static_cast<FloatType*>(acc_malloc(numberOfBytes));
        forces     = static_cast<FloatType*>(acc_malloc(numberOfBytes));
        oldForces  = static_cast<FloatType*>(acc_malloc(numberOfBytes));

        acc_memcpy_to_device(positions, positionsHost.data(), numberOfBytes);
        acc_memcpy_to_device(velocities, velocitiesHost.data(), numberOfBytes);
        acc_memcpy_to_device(forces, forcesHost.data(), numberOfBytes);

        // Zero the oldForces device buffer via a zero-filled host buffer.
        // (acc_memcpy_to_device takes a non-const source pointer, so this cannot be const.)
        std::vector<FloatType> zeros(n3, 0.0);
        acc_memcpy_to_device(oldForces, zeros.data(), numberOfBytes);
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
        const size_t numberOfBytes = n3 * sizeof(FloatType);

        // Copy device data back into the host mirrors
        acc_memcpy_from_device(positionsHost.data(), positions, numberOfBytes);
        acc_memcpy_from_device(velocitiesHost.data(), velocities, numberOfBytes);
        acc_memcpy_from_device(forcesHost.data(), forces, numberOfBytes);

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
        // Copy the global force into a plain local array. Indexing a std::array inside the
        // device region pulls in libstdc++'s hardened operator[] (std::__glibcxx_assert_fail),
        // which nvlink cannot resolve for device code on newer libstdc++ (e.g. NVHPC 26.3).
        const FloatType globalForce[3] = {
            _config.globalForce[0], _config.globalForce[1], _config.globalForce[2]
        };
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