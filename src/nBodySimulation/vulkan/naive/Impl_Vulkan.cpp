#include "Impl_Vulkan.h"

#include "KernelForce.h"
#include "KernelPosition.h"
#include "KernelVelocity.h"
#include "common/UtilityFloatArithmetic.h"

namespace ppb {

    template <typename FloatType>
    ImplVulkan<FloatType>::ImplVulkan(const ParticleSimulationConfig<FloatType> &config)
        : _config{config}
        , _timings{}
        , _manager{}
        , _sequence{_manager.sequence()}
        , _kernelForce{KERNELFORCE_COMP_SPV.begin(), KERNELFORCE_COMP_SPV.end()}
        , _kernelVelocity{KERNELVELOCITY_COMP_SPV.begin(), KERNELVELOCITY_COMP_SPV.end()}
        , _kernelPosition{KERNELPOSITION_COMP_SPV.begin(), KERNELPOSITION_COMP_SPV.end()}
    {}


    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplVulkan<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<float> positionsHost(particles.size() * 3, 0.0);
        std::vector<float> velocitiesHost(particles.size() * 3, 0.0);
        std::vector<float> forcesHost(particles.size() * 3, 0.0);
        std::vector<float> oldForcesHost(particles.size() * 3, 0.0);

        for (size_t i = 0; i < particles.size() * 3; ++i) {
            const size_t particleIndex = i / 3;
            const size_t componentIndex = i % 3;
            positionsHost[i] = particles[particleIndex].getPosition()[componentIndex];
            velocitiesHost[i] = particles[particleIndex].getVelocity()[componentIndex];
            forcesHost[i] = particles[particleIndex].getForce()[componentIndex];
            oldForcesHost[i] = 0.0;
        }
        
        auto positions = _manager.tensor(positionsHost);
        auto velocities = _manager.tensor(velocitiesHost);
        auto forces = _manager.tensor(forcesHost);
        auto oldForces = _manager.tensor(oldForcesHost);
        std::vector<std::shared_ptr<kp::Tensor>> params = {positions, velocities, forces, oldForces};
        _sequence->template record<kp::OpTensorSyncDevice>(params)->eval();
        _timings.reset();
        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            updatePositionsAndResetForce({positions, velocities, forces, oldForces});
            computeForces({positions, forces});
            updateVelocities({velocities, forces, oldForces});
        }
        _sequence->template record<kp::OpTensorSyncLocal>(params)->eval();

        positionsHost = positions->vector();
        velocitiesHost = velocities->vector();
        forcesHost = forces->vector();

        std::vector<Particle<float>> particlesRet{particles};
        for (size_t i = 0; i < particlesRet.size(); ++i) {
            particlesRet[i].setPosition({positionsHost[i * 3], positionsHost[i * 3 + 1], positionsHost[i * 3 + 2]});
            particlesRet[i].setVelocity({velocitiesHost[i * 3], velocitiesHost[i * 3 + 1], velocitiesHost[i * 3 + 2]});
            particlesRet[i].setForce({forcesHost[i * 3], forcesHost[i * 3 + 1], forcesHost[i * 3 + 2]});
        }

        return std::make_pair(particlesRet, _timings);
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::printBuffer(const std::shared_ptr<kp::Tensor> &buffer, bool floats) {

        _sequence->record<kp::OpTensorSyncLocal>({ buffer })->eval();

        std::cout << "BUFFER: ";
        
        if (floats) {
            auto data = buffer->vector<FloatType>();
            for (auto &v : data) std::cout << v << " ";
        } else {
            auto data = buffer->vector<uint>();
            for (auto &v : data) std::cout << v << " ";
        }

        std::cout << " END OF BUFFER" << std::endl;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::updatePositionsAndResetForce(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        constexpr unsigned int TILE_SIZE = 32;
        const unsigned int groups = util::ceilDiv<unsigned int>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        std::vector<float> pushConstants{_config.globalForce[0], _config.globalForce[1], _config.globalForce[2], _config.deltaT, *reinterpret_cast<float*>(&_config.size)};

        auto algorithm = _manager.algorithm(params, _kernelPosition, workgroup, {}, pushConstants);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushConstants)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.positionUpdateForceResetTime += elapsed_nanoseconds;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::updateVelocities(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        constexpr unsigned int TILE_SIZE = 32;
        const unsigned int groups = util::ceilDiv<unsigned int>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};
        std::vector<float> pushConstants({_config.deltaT, *reinterpret_cast<float*>(&_config.size)});

        auto algorithm = _manager.algorithm(params, _kernelVelocity, workgroup, {}, pushConstants);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushConstants)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.velocityUpdateTime += elapsed_nanoseconds;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::computeForces(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        constexpr unsigned int TILE_SIZE = 32;
        const unsigned int groups = util::ceilDiv<unsigned int>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        std::vector<unsigned int> pushConstants{static_cast<unsigned int>(_config.size)};

        auto algorithm = _manager.algorithm(params, _kernelForce, workgroup, {}, pushConstants);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushConstants)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.forceUpdateTime += elapsed_nanoseconds;
    }

    template class ImplVulkan<float>;

} // namespace ppb


