#include "Impl_Vulkan.h"
#include "Impl_Vulkan_PushConstants.h"

#include "KernelForce.h"
#include "KernelPosition.h"
#include "KernelVelocity.h"
#include "common/UtilityFloatArithmetic.h"

namespace ppb {

    template <typename FloatType>
    static uint kernel_calls(const ParticleSimulationConfig<FloatType> &config) {
        uint number_kernels = 3;
        int iterations = config.numberTimeSteps;
        return number_kernels * iterations;
    }

    template <typename FloatType>
    ImplVulkan<FloatType>::ImplVulkan(const ParticleSimulationConfig<FloatType> &config)
        : _config{config}
        , _timings{}
        , _manager{}
        , _sequence{_manager.sequence(config.use_kompute_timestamps ? kernel_calls(config) + 1 : 0)} // pass the number of kernel executions +1 to _manager.sequence, so that timestamps are collected WARNING: If the passed number is too small the program will stall
        , _kernelForce{KERNELFORCE_COMP_SPV.begin(), KERNELFORCE_COMP_SPV.end()}
        , _kernelVelocity{KERNELVELOCITY_COMP_SPV.begin(), KERNELVELOCITY_COMP_SPV.end()}
        , _kernelPosition{KERNELPOSITION_COMP_SPV.begin(), KERNELPOSITION_COMP_SPV.end()}
    {}


    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplVulkan<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<float> positionsHost(particles.size() * 4, 0.0);
        std::vector<float> velocitiesHost(particles.size() * 4, 0.0);
        std::vector<float> forcesHost(particles.size() * 4, 0.0);
        std::vector<float> oldForcesHost(particles.size() * 4, 0.0);

        for (size_t p = 0; p < particles.size(); ++p) {
            uint base = 4 * p;
            positionsHost[base] = particles[p].getPosition()[0];
            positionsHost[base + 1] = particles[p].getPosition()[1];
            positionsHost[base + 2] = particles[p].getPosition()[2];

            velocitiesHost[base] = particles[p].getVelocity()[0];
            velocitiesHost[base + 1] = particles[p].getVelocity()[1];
            velocitiesHost[base + 2] = particles[p].getVelocity()[2];

            forcesHost[base] = particles[p].getForce()[0];
            forcesHost[base + 1] = particles[p].getForce()[1];
            forcesHost[base + 2] = particles[p].getForce()[2];
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
            size_t base = i * 4;
            particlesRet[i].setPosition({positionsHost[base], positionsHost[base + 1], positionsHost[base + 2]});
            particlesRet[i].setVelocity({velocitiesHost[base], velocitiesHost[base + 1], velocitiesHost[base + 2]});
            particlesRet[i].setForce({forcesHost[base], forcesHost[base + 1], forcesHost[base + 2]});
        }

        return std::make_pair(particlesRet, _timings);
    }

    template <typename FloatType>
    long ImplVulkan<FloatType>::retrieve_timestamps() {
        std::vector<std::uint64_t> timestamps = _sequence->eval()->getTimestamps();
        if (timestamps.size() > 1) {
            long dt_ticks = timestamps[timestamps.size() - 1] - timestamps[timestamps.size() - 2];
            return dt_ticks;
        }
        return 0;
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
        constexpr unsigned int TILE_SIZE = _config.TILE_SIZE;
        const unsigned int groups = util::ceilDiv<unsigned int>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        PushPos pc{
            _config.globalForce[0],
            _config.globalForce[1],
            _config.globalForce[2],
            _config.deltaT,
            static_cast<uint32_t>(_config.size)
        };

        std::vector<uint32_t> pushData((sizeof(PushPos) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushPos));

        auto algorithm = _manager.algorithm(params, _kernelPosition, workgroup, {}, pushData);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushData)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        
        if (_config.use_kompute_timestamps) {
            _timings.positionUpdateForceResetTime += retrieve_timestamps();
        }
        else {
            _timings.positionUpdateForceResetTime += elapsed_nanoseconds;
        }
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::updateVelocities(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        constexpr unsigned int TILE_SIZE = _config.TILE_SIZE;
        const unsigned int groups = util::ceilDiv<unsigned int>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};
        
        PushVel pc{
            _config.deltaT,
            static_cast<uint32_t>(_config.size)
        };

        std::vector<uint32_t> pushData((sizeof(PushVel) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushVel));

        auto algorithm = _manager.algorithm(params, _kernelVelocity, workgroup, {}, pushData);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushData)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        
        if (_config.use_kompute_timestamps) {
            _timings.velocityUpdateTime += retrieve_timestamps();
        }
        else {
            _timings.velocityUpdateTime += elapsed_nanoseconds;
        }
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::computeForces(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        constexpr unsigned int TILE_SIZE = _config.TILE_SIZE;
        const unsigned int groups = util::ceilDiv<unsigned int>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        PushFor pc{
            static_cast<uint32_t>(_config.size),
        };

        std::vector<uint32_t> pushData((sizeof(PushFor) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushFor));

        auto algorithm = _manager.algorithm(params, _kernelForce, workgroup, {}, pushData);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushData)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        
        if (_config.use_kompute_timestamps) {
            _timings.forceUpdateTime += retrieve_timestamps();
        }
        else {
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }
    }

    template class ImplVulkan<float>;

} // namespace ppb
