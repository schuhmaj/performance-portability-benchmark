#include "Impl_Vulkan.h"

#include "KernelForce.h"
#include "KernelPosition.h"
#include "KernelVelocity.h"
#include "KernelCountNeighbors.h"
#include "KernelBlellochScan.h"
#include "KernelBlockSum.h"
#include "KernelVerlet.h"
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

        , _kernelCountNeighbors{KERNELCOUNTNEIGHBORS_COMP_SPV.begin(), KERNELCOUNTNEIGHBORS_COMP_SPV.end()}
        , _kernelBlellochScan{KERNELBLELLOCHSCAN_COMP_SPV.begin(), KERNELBLELLOCHSCAN_COMP_SPV.end()}
        , _kernelBlockSum{KERNELBLOCKSUM_COMP_SPV.begin(), KERNELBLOCKSUM_COMP_SPV.end()}
        , _kernelVerlet{KERNELVERLET_COMP_SPV.begin(), KERNELVERLET_COMP_SPV.end()}
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

        uint TILE_SIZE = _config.TILE_SIZE;
        uint nBlocks = (particles.size() + TILE_SIZE) / TILE_SIZE;
        uint neighborsLength = nBlocks * TILE_SIZE;

        std::vector<uint> neighborsHost(neighborsLength, 0);
        
        auto positions = _manager.tensor(positionsHost);
        auto velocities = _manager.tensor(velocitiesHost);
        auto forces = _manager.tensor(forcesHost);
        auto oldForces = _manager.tensor(oldForcesHost);

        auto neighbors = _manager.tensor(neighborsHost);

        std::vector<std::shared_ptr<kp::Tensor>> params = {positions, velocities, forces, oldForces, neighbors};
        _sequence->template record<kp::OpTensorSyncDevice>(params)->eval();
        _timings.reset();

        std::shared_ptr<kp::Tensor> verletList;

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            // here 10 is a magic number and should still be experimentally determined.
            if (i % 10 == 0) {
                countNeighbors({positions, neighbors});
                exclusiveScanBlelloch(neighbors, neighborsHost.size());
                verletList = createVerletList(positions, neighbors);
            }

            updatePositionsAndResetForce({positions, velocities, forces, oldForces});
            computeForces({positions, forces, verletList, neighbors});
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
    void ImplVulkan<FloatType>::countNeighbors(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        struct PushCount {
            uint32_t numParticles;
            float radius;
        };

        PushCount pc{
            static_cast<uint32_t>(_config.size),
            static_cast<float>(_config.influenceRadius)
        };

        std::vector<uint32_t> pushData((sizeof(PushCount) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushCount));

        auto algorithm = _manager.algorithm(params, _kernelCountNeighbors, workgroup, {}, pushData);

        // dispatch shader
        const auto start = std::chrono::high_resolution_clock::now();
        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData);
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsedNanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.neighborSearch += elapsedNanoseconds;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::exclusiveScanBlelloch(const std::shared_ptr<kp::Tensor> &data, const uint totalLength) {
        uint TILE_SIZE = _config.TILE_SIZE;
        // init blockSum buffer
        uint nBlocks = (totalLength + TILE_SIZE - 1) / TILE_SIZE;
        uint blockSumSize = ((nBlocks + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
        std::vector<uint> h(blockSumSize, 0);
        auto blockSum = _manager.tensor(h);

        kp::Workgroup workgroup{{nBlocks, 1, 1}};
        std::vector<uint32_t> pushData{totalLength, TILE_SIZE};

        auto algorithmBlelloch = _manager.algorithm({data, blockSum}, _kernelBlellochScan, workgroup, {}, pushData);

        // dispatch shader
        const auto startBlelloch = std::chrono::high_resolution_clock::now();
        _sequence->template record<kp::OpAlgoDispatch>(algorithmBlelloch, pushData);
        const auto endBlelloch = std::chrono::high_resolution_clock::now();

        const double elapsedNanosecondsBlelloch =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endBlelloch - startBlelloch).count());

        _timings.neighborSearch += elapsedNanosecondsBlelloch;

        // calculate prefix sum of block sum
        if (nBlocks > 1) {
            exclusiveScanBlelloch(blockSum, blockSumSize);

            // add block offset
            std::vector<uint32_t> pushData{totalLength};

            auto algorithmBlock = _manager.algorithm({data, blockSum}, _kernelBlockSum, workgroup, {}, pushData);

            // dispatch shader
            const auto startBlock = std::chrono::high_resolution_clock::now();
            _sequence->template record<kp::OpAlgoDispatch>(algorithmBlock, pushData);
            const auto endBlock = std::chrono::high_resolution_clock::now();

            const double elapsedNanosecondsBlock =
                static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endBlock - startBlock).count());

            _timings.neighborSearch += elapsedNanosecondsBlock;
        }
    }

    template <typename FloatType>
    std::shared_ptr<kp::Tensor> ImplVulkan<FloatType>::createVerletList(const std::shared_ptr<kp::Tensor> &positions, const std::shared_ptr<kp::Tensor> &neighborsStarts) {
        _sequence->record<kp::OpTensorSyncLocal>({ neighborsStarts })->eval();
        auto data = neighborsStarts->vector<uint>();
        uint nNeighbors = std::max(data[_config.size], 1u);

        std::vector<uint> verletListHost(nNeighbors, 0);
        auto verletList = _manager.tensor(verletListHost);

        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        struct PushVerlet {
            uint32_t numParticles;
            float radius;
        };

        PushVerlet pc{
            static_cast<uint32_t>(_config.size),
            static_cast<float>(_config.influenceRadius)
        };

        std::vector<uint32_t> pushData((sizeof(PushVerlet) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushVerlet));

        auto algorithm = _manager.algorithm({positions, verletList, neighborsStarts}, _kernelVerlet, workgroup, {}, pushData);

        // dispatch shader
        const auto start = std::chrono::high_resolution_clock::now();
        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData);
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsedNanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.neighborSearch += elapsedNanoseconds;

        return verletList;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::updatePositionsAndResetForce(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        constexpr unsigned int TILE_SIZE = _config.TILE_SIZE;
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
        constexpr unsigned int TILE_SIZE = _config.TILE_SIZE;
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
        constexpr unsigned int TILE_SIZE = _config.TILE_SIZE;
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


