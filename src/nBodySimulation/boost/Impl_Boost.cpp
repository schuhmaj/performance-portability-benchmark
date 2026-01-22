#include "Impl_Boost.h"
#include "common/UtilityFloatArithmetic.h"

namespace ppb {

    template <typename FloatType>
    BoostParticleSoA<FloatType>::BoostParticleSoA(const std::vector<Particle<FloatType>> &ref, const boost::compute::context &context, boost::compute::command_queue &queue)
        : _ref{ref}
        , positions{ref.size(), context}
        , velocities{ref.size(), context}
        , forces{ref.size(), context}
        , oldForces{ref.size(), context}
        , positionsHost{ref.size()}
        , velocitiesHost{ref.size()}
        , forcesHost{ref.size()}
        , queue{queue}
    {
        const auto transformFloat4 = [](auto getter) {
            return [getter](const auto& particle) {
                const auto arr = std::invoke(getter, particle);
                return boost::compute::float4_(arr[0], arr[1], arr[2], 0.0f);
            };
        };

        std::transform(ref.begin(), ref.end(), positionsHost.begin(), transformFloat4(&Particle<FloatType>::getPosition));
        std::transform(ref.begin(), ref.end(), velocitiesHost.begin(), transformFloat4(&Particle<FloatType>::getVelocity));
        std::transform(ref.begin(), ref.end(), forcesHost.begin(), transformFloat4(&Particle<FloatType>::getForce));

        boost::compute::copy(positionsHost.begin(), positionsHost.end(), positions.begin(), queue);
        boost::compute::copy(velocitiesHost.begin(), velocitiesHost.end(), velocities.begin(), queue);
        boost::compute::copy(forcesHost.begin(), forcesHost.end(), forces.begin(), queue);
        boost::compute::fill(oldForces.begin(), oldForces.end(), boost::compute::float4_(0.0f, 0.0f, 0.0f, 0.0f), queue);
    }

    template <typename FloatType>
    std::vector<Particle<FloatType>> BoostParticleSoA<FloatType>::toParticles() {
        std::vector<Particle<FloatType>> particles{_ref};
        boost::compute::copy(positions.begin(), positions.end(), positionsHost.begin(), queue);
        boost::compute::copy(velocities.begin(), velocities.end(), velocitiesHost.begin(), queue);
        boost::compute::copy(forces.begin(), forces.end(), forcesHost.begin(), queue);
        const auto toArray = [](const boost::compute::float4_ &float4) -> std::array<FloatType, 3> {
            return {float4.x, float4.y, float4.z};
        };
        for (size_t i = 0; i < particles.size(); ++i) {
            particles[i].setPosition(toArray(positionsHost[i]));
            particles[i].setVelocity(toArray(velocitiesHost[i]));
            particles[i].setForce(toArray(forcesHost[i]));
        }
        return particles;
    }


    template <typename FloatType>
    ImplBoost<FloatType>::ImplBoost(const ParticleSimulationConfig<FloatType> &config)
        : _config{config}
        , _numParticles{static_cast<unsigned int>(config.size)}
        , _deltaT{config.deltaT}
        , _globalForce{_config.globalForce[0], _config.globalForce[1], _config.globalForce[2], 0.0f}
        , gpu{boost::compute::system::default_device()}
        , context{gpu}
        , queue{context, gpu, boost::compute::command_queue::enable_profiling}
        , program{boost::compute::program::build_with_source(std::string(KERNEL_SOURCE), context)}
        , kernelPositionUpdate{program, std::string("update_positions_reset_forces")}
        , kerneVelocityUpdate{program, std::string("update_velocities")}
        , kernelForceUpdate{program, std::string("compute_forces")}
    {}

    template <typename FloatType>
    void ImplBoost<FloatType>::init(const std::vector<Particle<FloatType>> &particles) {
        _timings.reset();
        _particles.emplace(particles, context, queue);
        kernelPositionUpdate.set_arg(0, _particles->positions);
        kernelPositionUpdate.set_arg(1, _particles->velocities);
        kernelPositionUpdate.set_arg(2, _particles->forces);
        kernelPositionUpdate.set_arg(3, _particles->oldForces);
        kernelPositionUpdate.set_arg(4, _globalForce);
        kernelPositionUpdate.set_arg(5, _deltaT);
        kernelPositionUpdate.set_arg(6, _numParticles);

        kerneVelocityUpdate.set_arg(0, _particles->velocities);
        kerneVelocityUpdate.set_arg(1, _particles->forces);
        kerneVelocityUpdate.set_arg(2, _particles->oldForces);
        kerneVelocityUpdate.set_arg(3, _deltaT);
        kerneVelocityUpdate.set_arg(4, _numParticles);

        kernelForceUpdate.set_arg(0, _particles->positions);
        kernelForceUpdate.set_arg(1, _particles->forces);
        kernelForceUpdate.set_arg(2, _numParticles);
    }


    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplBoost<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        init(particles);
        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce();
            computeForces();
            updateVelocities();
        }
        return std::make_pair(_particles->toParticles(), _timings);
    }

    template <typename FloatType>
    void ImplBoost<FloatType>::updatePositionsAndResetForce() {
        const size_t localSize = 32;
        const size_t globalSize = util::roundUp<size_t>(_numParticles, localSize);

        const auto event = queue.enqueue_nd_range_kernel(
            kernelPositionUpdate, 1, nullptr, &globalSize, &localSize
        );
        event.wait();
        _timings.positionUpdateForceResetTime += event.duration<std::chrono::nanoseconds>().count();
    }

    template <typename FloatType>
    void ImplBoost<FloatType>::updateVelocities() {
        const size_t localSize = 32;
        const size_t globalSize = util::roundUp<size_t>(_numParticles, localSize);

        const auto event = queue.enqueue_nd_range_kernel(
            kerneVelocityUpdate, 1, nullptr, &globalSize, &localSize
        );
        event.wait();
        _timings.velocityUpdateTime += event.duration<std::chrono::nanoseconds>().count();
    }

    template <typename FloatType>
    void ImplBoost<FloatType>::computeForces() {
        const size_t localSize[2] = {32, 32};
        const size_t globalSize[2] = {
            util::roundUp<size_t>(_numParticles, localSize[0]),
            util::roundUp<size_t>(_numParticles, localSize[1])
        };

        const auto event = queue.enqueue_nd_range_kernel(kernelForceUpdate, 1, nullptr, globalSize, localSize);
        event.wait();
        _timings.forceUpdateTime += event.duration<std::chrono::nanoseconds>().count();
    }

    template class ImplBoost<float>;
    template class BoostParticleSoA<float>;

} // namespace ppb


