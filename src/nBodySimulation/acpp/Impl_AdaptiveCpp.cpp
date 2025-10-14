#include "Impl_AdaptiveCpp.h"
#include <sycl/sycl.hpp>
#include <span>

namespace ppb {

    template <typename FloatType>
    ImplAdaptiveCpp<FloatType>::ImplAdaptiveCpp(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::vector<Particle<FloatType>> ImplAdaptiveCpp<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<Particle<FloatType>> particlesCopy{particles};

        sycl::queue queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

        size_t size = particlesCopy.size();
        ParticleContainer<FloatType> particle_container(particles);

        FloatType *positionsUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(4 * sizeof(FloatType), 4 * size * sizeof(FloatType), queue));
        FloatType *velocitiesUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(4 * sizeof(FloatType), 4 * size * sizeof(FloatType), queue));
        FloatType *forcesUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(4 * sizeof(FloatType), 4 * size * sizeof(FloatType), queue));
        FloatType *oldForcesUSM = static_cast<FloatType *>(sycl::aligned_alloc_device(4 * sizeof(FloatType), 4 * size * sizeof(FloatType), queue));

        if (!positionsUSM || !velocitiesUSM || !forcesUSM || !oldForcesUSM) {
            throw std::runtime_error("USM allocation faild");
        }

        std::memcpy(positionsUSM, particle_container.getPositions(), 4 * size * sizeof(FloatType));
        std::memcpy(velocitiesUSM, particle_container.getVelocities(), 4 * size * sizeof(FloatType));
        std::memcpy(forcesUSM, particle_container.getForces(), 4 * size * sizeof(FloatType));
        std::memcpy(oldForcesUSM, particle_container.getOldForces(), 4 * size * sizeof(FloatType));

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce(queue, positionsUSM, velocitiesUSM, forcesUSM, oldForcesUSM, size);
            computeForces_atomic(queue, positionsUSM, forcesUSM, size);
            updateVelocities(queue, velocitiesUSM, forcesUSM, oldForcesUSM, size);
        }

        particle_container.extractParticleData(particlesCopy);

        std::memcpy(particle_container.getPositions(), positionsUSM, 4 * size * sizeof(FloatType));
        std::memcpy(particle_container.getVelocities(), velocitiesUSM, 4 * size * sizeof(FloatType));
        std::memcpy(particle_container.getForces(), forcesUSM, 4 * size * sizeof(FloatType));
        std::memcpy(particle_container.getOldForces(), oldForcesUSM, 4 * size * sizeof(FloatType));

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
            std::span<FloatType, 3> p(&(positionsUSM[3 * idx]), 3);
            std::span<FloatType, 3> v(&(velocitiesUSM[3 * idx]), 3);
            std::span<FloatType, 3> f(&(forcesUSM[3 * idx]), 3);
            std::span<FloatType, 3> oldF(&(oldForcesUSM[3 * idx]), 3);
            
            std::copy(f.begin(), f.end(), oldF.begin());
            std::copy(globalForce.begin(), globalForce.end(), f.begin());

            v[0] *= deltaT;
            v[1] *= deltaT;
            v[2] *= deltaT;
            f[0] *= (deltaT * deltaT / (2 * m));
            f[1] *= (deltaT * deltaT / (2 * m));
            f[2] *= (deltaT * deltaT / (2 * m));
            p[0] += v[0] + f[0];
            p[1] += v[1] + f[1];
            p[2] += v[2] + f[2];
        });
        queue.wait();
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updateVelocities(sycl::queue &queue, FloatType *velocitiesUSM, FloatType *forcesUSM, FloatType *oldForcesUSM, const size_t &size) {
        using namespace ppb::util;

        const auto deltaT = _config.deltaT;

        queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            const FloatType m = FloatType(1.0);
            std::span<FloatType, 3> v(&(velocitiesUSM[3 * idx]), 3);
            const std::span<FloatType, 3> f(&(forcesUSM[3 * idx]), 3);
            const std::span<FloatType, 3> oldF(&(oldForcesUSM[3 * idx]), 3);

            v[0] += (f[0] + oldF[0]) * (deltaT / (2 * m));
            v[1] += (f[1] + oldF[1]) * (deltaT / (2 * m));
            v[2] += (f[2] + oldF[2]) * (deltaT / (2 * m));
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

            if (i == j) return;

            const FloatType sigma = FloatType(1.0);
            const FloatType sigmaSquared = sigma * sigma;
            const FloatType epsilon = FloatType(5.0);
            const FloatType epsilon24 = epsilon * FloatType(24);

            const std::span<FloatType, 3> pi(&(positionsUSM[3 * i]), 3);
            const std::span<FloatType, 3> pj(&(positionsUSM[3 * j]), 3);

            std::array<FloatType, 3> dr{pi[0], pi[1], pi[2]};
            dr[0] -= pj[0];
            dr[1] -= pj[1];
            dr[2] -= pj[2];
            const FloatType dr2 = dot(dr, dr);

            const FloatType invdr2 = FloatType(1.0) / dr2;
            FloatType lj6 = sigmaSquared * invdr2;
            lj6 = lj6 * lj6 * lj6;
            const FloatType lj12 = lj6 * lj6;
            const FloatType lj12m6 = lj12 - lj6;
            const FloatType fac = epsilon24 * (lj12 + lj12m6) * invdr2;
            const std::array<FloatType, 3> f = dr * fac;

            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi0(forcesUSM[3 * i + 0]);
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi1(forcesUSM[3 * i + 1]);
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi2(forcesUSM[3 * i + 2]);
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


