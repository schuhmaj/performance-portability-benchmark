#include "Impl_AdaptiveCpp.h"
#include <sycl/sycl.hpp>

namespace ppb {

    template <typename FloatType>
    ImplAdaptiveCpp<FloatType>::ImplAdaptiveCpp(const ParticleSimulationConfig<FloatType> &config) : _config{config} {}

    template <typename FloatType>
    std::vector<Particle<FloatType>> ImplAdaptiveCpp<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<Particle<FloatType>> particlesCopy{particles};

        sycl::queue queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

        const size_t size = particlesCopy.size();
        Particle<FloatType> *particlesUSM = sycl::malloc_shared<Particle<FloatType>>(size, queue);
        if (!particlesUSM) {
            throw std::runtime_error("USM allocation faild");
        }

        std::memcpy(particlesUSM, particlesCopy.data(), size * sizeof(Particle<FloatType>));

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce(queue, particlesUSM, size);
            computeForces_atomic(queue, particlesUSM, size);
            updateVelocities(queue, particlesUSM, size);
        }

        std::memcpy(particlesCopy.data(), particlesUSM, size * sizeof(Particle<FloatType>));

        sycl::free(particlesUSM, queue);

        return particlesCopy;
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updatePositionsAndResetForce(sycl::queue &queue, Particle<FloatType> *particlesUSM, const size_t &size) {
        using namespace ppb::util;

        const auto globalForce = _config.globalForce;
        const auto deltaT = _config.deltaT;

        queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            auto &particle = particlesUSM[idx];
            
            const auto m = particle.getMass();
            auto v = particle.getVelocity();
            auto f = particle.getForce();
            particle.setOldForce(f);
            particle.setForce(globalForce);
            v *= deltaT;
            f *= (deltaT * deltaT / (2 * m));
            const auto displacement = v + f;
            particle.addPosition(displacement);
        });
        queue.wait();
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::updateVelocities(sycl::queue &queue, Particle<FloatType> *particlesUSM, const size_t &size) {
        using namespace ppb::util;

        const auto deltaT = _config.deltaT;

        queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            auto &particle = particlesUSM[idx];

            const auto m = particle.getMass();
            const auto force = particle.getForce();
            const auto oldForce = particle.getOldForce();
            const auto changeinVel = (force + oldForce) * (deltaT / (2 * m));
            particle.addVelocity(changeinVel);
        });
        queue.wait();
    }

    template <typename FloatType>
    void ImplAdaptiveCpp<FloatType>::computeForces_atomic(sycl::queue &queue, Particle<FloatType> *particlesUSM, const size_t &size) {
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

            if (i >= size || j >= size) return;
            if (i == j) return;

            auto &pi = particlesUSM[i];
            const auto &pj = particlesUSM[j];

            const auto sigma = (pi.getSigma() + pj.getSigma()) * FloatType(0.5);
            const auto sigmaSquared = sigma * sigma;
            const auto epsilon24 = sycl::sqrt(pi.getEpsilon() * pj.getEpsilon()) * FloatType(24);

            const auto dr = pi.getPosition() - pj.getPosition();
            const auto dr2 = dot(dr, dr);

            const auto invdr2 = FloatType(1) / dr2;
            auto lj6 = sigmaSquared * invdr2;
            lj6 = lj6 * lj6 * lj6;
            const auto lj12 = lj6 * lj6;
            const auto lj12m6 = lj12 - lj6;
            const auto fac = epsilon24 * (lj12 + lj12m6) * invdr2;
            const auto f = dr * fac;
        
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi0(pi.getForce()[0]);
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi1(pi.getForce()[1]);
            sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi2(pi.getForce()[2]);
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


