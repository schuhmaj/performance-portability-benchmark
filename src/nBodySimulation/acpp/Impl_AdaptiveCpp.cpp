#include "Impl_AdaptiveCpp.h"
#include <sycl/sycl.hpp>
#include <span>

namespace ppb {

    template <typename FloatType>
    ImplAdaptiveCpp<FloatType>::ImplAdaptiveCpp(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::vector<Particle<FloatType>> ImplAdaptiveCpp<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<Particle<FloatType>> particlesCopy = particles;

        sycl::queue queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

        size_t size = particlesCopy.size();

        // build SoA from AoO
        ParticleContainer<FloatType> particle_container(particles);

        // set up memory on device
        FloatType *positionsUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(sizeof(FloatType), 4 * size * sizeof(FloatType), queue));
        FloatType *velocitiesUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(sizeof(FloatType), 4 * size * sizeof(FloatType), queue));
        FloatType *forcesUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(sizeof(FloatType), 4 * size * sizeof(FloatType), queue));
        FloatType *oldForcesUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(sizeof(FloatType), 4 * size * sizeof(FloatType), queue));

        if (!positionsUSM || !velocitiesUSM || !forcesUSM || !oldForcesUSM) {
            throw std::runtime_error("USM allocation faild");
        }

        // move data to device
        queue.memcpy(positionsUSM, particle_container.getPositions(), 4 * size * sizeof(FloatType));
        queue.memcpy(velocitiesUSM, particle_container.getVelocities(), 4 * size * sizeof(FloatType));
        queue.memcpy(forcesUSM, particle_container.getForces(), 4 * size * sizeof(FloatType));
        queue.memcpy(oldForcesUSM, particle_container.getOldForces(), 4 * size * sizeof(FloatType));

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce(queue, positionsUSM, velocitiesUSM, forcesUSM, oldForcesUSM, size);
            computeForces_atomic(queue, positionsUSM, forcesUSM, size);
            updateVelocities(queue, velocitiesUSM, forcesUSM, oldForcesUSM, size);
        }

        // move data to host
        queue.memcpy(particle_container.getPositions(), positionsUSM, 4 * size * sizeof(FloatType));
        queue.memcpy(particle_container.getVelocities(), velocitiesUSM, 4 * size * sizeof(FloatType));
        queue.memcpy(particle_container.getForces(), forcesUSM, 4 * size * sizeof(FloatType));
        queue.memcpy(particle_container.getOldForces(), oldForcesUSM, 4 * size * sizeof(FloatType));

        // convert SoA to AoO
        particle_container.extractParticleData(particlesCopy);

        sycl::free(positionsUSM, queue);
        sycl::free(velocitiesUSM, queue);
        sycl::free(forcesUSM, queue);
        sycl::free(oldForcesUSM, queue);

        return particlesCopy;
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updatePositionsAndResetForce(sycl::queue &queue, FloatType *positionsUSM, FloatType *velocitiesUSM, FloatType *forcesUSM, FloatType *oldForcesUSM, const size_t &size) {
        using namespace ppb::util;

        const auto globalForce = _config.globalForce;
        const auto deltaT = _config.deltaT;

        queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            const FloatType m = FloatType(1.0);
            
            // oldForce = force
            oldForcesUSM[4 * idx + 0] = forcesUSM[4 * idx + 0];
            oldForcesUSM[4 * idx + 1] = forcesUSM[4 * idx + 1];
            oldForcesUSM[4 * idx + 2] = forcesUSM[4 * idx + 2];

            // force = globalForce
            forcesUSM[4 * idx + 0] = globalForce[0];
            forcesUSM[4 * idx + 1] = globalForce[1];
            forcesUSM[4 * idx + 2] = globalForce[2];

            // velocityFactor = velocity * deltaT
            std::array<FloatType, 3> vfac{
                velocitiesUSM[4 * idx + 0] * deltaT,
                velocitiesUSM[4 * idx + 1] * deltaT,
                velocitiesUSM[4 * idx + 2] * deltaT
            };

            // forceFactor = force * deltaT ** 2 / (2 * mass)
            std::array<FloatType, 3> ffac{
                forcesUSM[4 * idx + 0] * (deltaT * deltaT / (2 * m)),
                forcesUSM[4 * idx + 1] * (deltaT * deltaT / (2 * m)),
                forcesUSM[4 * idx + 2] * (deltaT * deltaT / (2 * m))
            };

            // change in position = velocityFactor + forceFactor
            positionsUSM[4 * idx + 0] += vfac[0] + ffac[0];
            positionsUSM[4 * idx + 1] += vfac[1] + ffac[1];
            positionsUSM[4 * idx + 2] += vfac[2] + ffac[2];
        });
        queue.wait();
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updateVelocities(sycl::queue &queue, FloatType *velocitiesUSM, FloatType *forcesUSM, FloatType *oldForcesUSM, const size_t &size) {
        using namespace ppb::util;

        const auto deltaT = _config.deltaT;

        queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            const FloatType m = FloatType(1.0);

            // change in velocity = (force + oldForce) * deltaT / (2 * mass)
            velocitiesUSM[0] += (forcesUSM[0] + oldForcesUSM[0]) * (deltaT / (2 * m));
            velocitiesUSM[1] += (forcesUSM[1] + oldForcesUSM[1]) * (deltaT / (2 * m));
            velocitiesUSM[2] += (forcesUSM[2] + oldForcesUSM[2]) * (deltaT / (2 * m));
        });
        queue.wait();
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::computeForces_atomic(sycl::queue &queue, FloatType *positionsUSM, FloatType *forcesUSM, const size_t &size) {
        using namespace ppb::util;

        // tuned to best ratio
        constexpr size_t local_size_x = sizeof(FloatType) == 4 ? 512 : 64;
        constexpr size_t local_size_y = sizeof(FloatType) == 4 ? 2 : 16;

        const size_t global_size_x = ((size + local_size_x - 1) / local_size_x) * local_size_x;
        const size_t global_size_y = ((size + local_size_y - 1) / local_size_y) * local_size_y;

        sycl::range<2> global_range(global_size_x, global_size_y);
        sycl::range<2> local_range(local_size_x, local_size_y);
        sycl::nd_range<2> nd_range(global_range, local_range);

        queue.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            size_t i = item.get_global_id(0);
            size_t j = item.get_global_id(1);

            if (i == j || i >= size || j >= size) return;

            const FloatType sigma = FloatType(0.5);
            const FloatType sigmaSquared = sigma * sigma;
            const FloatType epsilon = FloatType(5.0);
            const FloatType epsilon24 = epsilon * FloatType(24);

            // distance = position_i - position_j
            std::array<FloatType, 3> dist{
                positionsUSM[4 * i + 0] - positionsUSM[4 * j + 0],
                positionsUSM[4 * i + 1] - positionsUSM[4 * j + 1],
                positionsUSM[4 * i + 2] - positionsUSM[4 * j + 2]
            };

            const FloatType distSquared = dot(dist, dist);

            const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
            FloatType lj6 = sigmaSquared * inverseDistSquared;
            lj6 = lj6 * lj6 * lj6;
            const FloatType lj12 = lj6 * lj6;
            const FloatType lj12m6 = lj12 - lj6;
            const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;
            const std::array<FloatType, 3> f = dist * fac;

            // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi0(forcesUSM[4 * i + 0]);
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi1(forcesUSM[4 * i + 1]);
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi2(forcesUSM[4 * i + 2]);
            atomic_fi0.fetch_add(f[0]);
            atomic_fi1.fetch_add(f[1]);
            atomic_fi2.fetch_add(f[2]);
        });
        queue.wait();
    }

    /* Explicit Instantiation for float and double */
    template class ImplAdaptiveCpp<float>;
    template class ImplAdaptiveCpp<double>;

} // namespace ppb


