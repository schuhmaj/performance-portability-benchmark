#include "Impl_Vulkan.h"
#include "Push_Constants.h"
#include "Common_Push_Constants.h"

#include "KernelForce.h"
#include "KernelPosition.h"
#include "KernelVelocity.h"
#include "KernelCountNeighbors.h"
#include "KernelBlellochScan.h"
#include "KernelBlockSum.h"
#include "KernelVerlet.h"
#include "common/UtilityFloatArithmetic.h"

namespace ppb {

    static uint blelloch_executions(uint length, uint TILE_SIZE) {
        uint total_calls = 1;
        uint nBlocks = (length + TILE_SIZE - 1) / TILE_SIZE;
        if (nBlocks > 1) {
            uint blockSumSize = ((nBlocks + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
            total_calls += blelloch_executions(blockSumSize, TILE_SIZE) + 1;
        }
        return total_calls;
    }

    template <typename FloatType>
    static uint kernel_calls(const ParticleSimulationConfig<FloatType> &config) {
        uint number_kernels_each_frame = 3;
        int iterations = config.numberTimeSteps;
        uint interval = config.interval_neighbor_search;

        uint TILE_SIZE = config.TILE_SIZE;
        uint nBlocks = (config.size + TILE_SIZE) / TILE_SIZE;
        uint neighborsLength = nBlocks * TILE_SIZE;
        uint number_kernels_neighbor_search = 3 + blelloch_executions(neighborsLength, TILE_SIZE);

        uint countNeighborSearch = util::ceilDiv<uint>(iterations, interval);

        return number_kernels_each_frame * iterations + countNeighborSearch * number_kernels_neighbor_search;
    }

    template <typename FloatType>
    ImplVulkan<FloatType>::ImplVulkan(const ParticleSimulationConfig<FloatType> &config)
        : _config{config}
        , _timings{}
        , _manager{}
        , _kernelForce{KERNELFORCE_COMP_SPV.begin(), KERNELFORCE_COMP_SPV.end()}
        , _kernelVelocity{KERNELVELOCITY_COMP_SPV.begin(), KERNELVELOCITY_COMP_SPV.end()}
        , _kernelPosition{KERNELPOSITION_COMP_SPV.begin(), KERNELPOSITION_COMP_SPV.end()}

        , _kernelCountNeighbors{KERNELCOUNTNEIGHBORS_COMP_SPV.begin(), KERNELCOUNTNEIGHBORS_COMP_SPV.end()}
        , _kernelBlellochScan{KERNELBLELLOCHSCAN_COMP_SPV.begin(), KERNELBLELLOCHSCAN_COMP_SPV.end()}
        , _kernelBlockSum{KERNELBLOCKSUM_COMP_SPV.begin(), KERNELBLOCKSUM_COMP_SPV.end()}
        , _kernelVerlet{KERNELVERLET_COMP_SPV.begin(), KERNELVERLET_COMP_SPV.end()}

        , _sequence{_manager.sequence(config.use_kompute_timestamps ? kernel_calls(config) + 1 : 0)}
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

        uint TILE_SIZE = ParticleSimulationConfig<FloatType>::TILE_SIZE;
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
            if (i % ParticleSimulationConfig<FloatType>::interval_neighbor_search == 0) {
                countNeighbors({positions, neighbors});
                exclusiveScanBlelloch(neighbors, neighborsHost.size());
                verletList = createVerletList(positions, neighbors);
            }

            updatePositionsAndResetForce({positions, velocities, forces, oldForces});
            computeForces({positions, forces, verletList, neighbors});
            updateVelocities({velocities, forces, oldForces});
        }
        _sequence->template record<kp::OpTensorSyncLocal>(params)->eval();

        //printBuffer(positions, true);

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
        std::vector<std::uint64_t> timestamps = _sequence->getTimestamps();
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
    void ImplVulkan<FloatType>::countNeighbors(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = ParticleSimulationConfig<FloatType>::TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        PushCount pc{
            static_cast<uint32_t>(_config.size),
            static_cast<float>(ParticleSimulationConfig<FloatType>::influenceRadius)
        };

        std::vector<uint32_t> pushData((sizeof(PushCount) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushCount));

        auto algorithm = _manager.algorithm(params, _kernelCountNeighbors, workgroup, {}, pushData);

        // dispatch shader
        const auto start = std::chrono::high_resolution_clock::now();
        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData)->eval();
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsedNanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ParticleSimulationConfig<FloatType>::use_kompute_timestamps) {
            _timings.neighborSearch += retrieve_timestamps();
        }
        else {
            _timings.neighborSearch += elapsedNanoseconds;
        }
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::exclusiveScanBlelloch(const std::shared_ptr<kp::Tensor> &data, const uint totalLength) {
        uint TILE_SIZE = ParticleSimulationConfig<FloatType>::TILE_SIZE;
        // init blockSum buffer
        uint nBlocks = (totalLength + TILE_SIZE - 1) / TILE_SIZE;
        uint blockSumSize = ((nBlocks + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
        std::vector<uint> h(blockSumSize, 0);
        auto blockSum = _manager.tensor(h);

        kp::Workgroup workgroup{{nBlocks, 1, 1}};

        PushBlelloch pc{
            totalLength,
            TILE_SIZE
        };
        std::vector<uint32_t> pushData((sizeof(PushBlelloch) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushBlelloch));

        auto algorithmBlelloch = _manager.algorithm({data, blockSum}, _kernelBlellochScan, workgroup, {}, pushData);

        // dispatch shader
        const auto startBlelloch = std::chrono::high_resolution_clock::now();
        _sequence->template record<kp::OpAlgoDispatch>(algorithmBlelloch, pushData)->eval();
        const auto endBlelloch = std::chrono::high_resolution_clock::now();

        const double elapsedNanosecondsBlelloch =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endBlelloch - startBlelloch).count());
        if (ParticleSimulationConfig<FloatType>::use_kompute_timestamps) {
            _timings.neighborSearch += retrieve_timestamps();
        }
        else {
            _timings.neighborSearch += elapsedNanosecondsBlelloch;
        }

        // calculate prefix sum of block sum
        if (nBlocks > 1) {
            exclusiveScanBlelloch(blockSum, blockSumSize);

            // add block offset
            PushBlock pc{
                totalLength
            };
            std::vector<uint32_t> pushData((sizeof(PushBlock) + 3) / 4);
            std::memcpy(pushData.data(), &pc, sizeof(PushBlock));

            auto algorithmBlock = _manager.algorithm({data, blockSum}, _kernelBlockSum, workgroup, {}, pushData);

            // dispatch shader
            const auto startBlock = std::chrono::high_resolution_clock::now();
            _sequence->template record<kp::OpAlgoDispatch>(algorithmBlock, pushData)->eval();
            const auto endBlock = std::chrono::high_resolution_clock::now();

            const double elapsedNanosecondsBlock =
                static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endBlock - startBlock).count());
            if (ParticleSimulationConfig<FloatType>::use_kompute_timestamps) {
                _timings.neighborSearch += retrieve_timestamps();   
            }
            else {
                _timings.neighborSearch += elapsedNanosecondsBlock; 
            }


        }
    }

    template <typename FloatType>
    std::shared_ptr<kp::Tensor> ImplVulkan<FloatType>::createVerletList(const std::shared_ptr<kp::Tensor> &positions, const std::shared_ptr<kp::Tensor> &neighborsStarts) {
        _sequence->record<kp::OpTensorSyncLocal>({ neighborsStarts })->eval();
        auto data = neighborsStarts->vector<uint>();
        uint nNeighbors = std::max(data[_config.size], 1u);

        std::vector<uint> verletListHost(nNeighbors, 0);
        auto verletList = _manager.tensor(verletListHost);

        uint TILE_SIZE = ParticleSimulationConfig<FloatType>::TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        PushVerlet pc{
            static_cast<uint32_t>(_config.size),
            static_cast<float>(ParticleSimulationConfig<FloatType>::influenceRadius)
        };
        std::vector<uint32_t> pushData((sizeof(PushVerlet) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushVerlet));

        auto algorithm = _manager.algorithm({positions, verletList, neighborsStarts}, _kernelVerlet, workgroup, {}, pushData);

        // dispatch shader
        const auto start = std::chrono::high_resolution_clock::now();
        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData)->eval();
        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsedNanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ParticleSimulationConfig<FloatType>::use_kompute_timestamps) {
            _timings.neighborSearch += retrieve_timestamps();
        }
        else {
            _timings.neighborSearch += elapsedNanoseconds;
        }

        return verletList;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::updatePositionsAndResetForce(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = ParticleSimulationConfig<FloatType>::TILE_SIZE;
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
        if (ParticleSimulationConfig<FloatType>::use_kompute_timestamps) {
            _timings.positionUpdateForceResetTime += retrieve_timestamps();
        }
        else {
            _timings.positionUpdateForceResetTime += elapsed_nanoseconds;
        }
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::updateVelocities(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = ParticleSimulationConfig<FloatType>::TILE_SIZE;
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
        if (ParticleSimulationConfig<FloatType>::use_kompute_timestamps) {
            _timings.velocityUpdateTime += retrieve_timestamps();
        }
        else {
            _timings.velocityUpdateTime += elapsed_nanoseconds;
        }
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::computeForces(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = ParticleSimulationConfig<FloatType>::TILE_SIZE;
        const unsigned int groups = util::ceilDiv<unsigned int>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        PushBlock pc{
            static_cast<uint32_t>(_config.size)
        };
        std::vector<uint32_t> pushData((sizeof(PushBlock) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushBlock));

        auto algorithm = _manager.algorithm(params, _kernelForce, workgroup, {}, pushData);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushData)->eval();

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        if (ParticleSimulationConfig<FloatType>::use_kompute_timestamps) {
            _timings.forceUpdateTime += retrieve_timestamps();
        }
        else {
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }
    }

    template class ImplVulkan<float>;

} // namespace ppb
