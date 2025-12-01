#include "Impl_Vulkan.h"

#include "KernelForce.h"
#include "KernelPosition.h"
#include "KernelVelocity.h"
#include "KernelHistogram.h"
#include "KernelBlellochScan.h"
#include "KernelBlockSum.h"
#include "KernelIdCells.h"
#include "common/UtilityFloatArithmetic.h"

#include <chrono>

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

        , _kernelHistogram{KERNELHISTOGRAM_COMP_SPV.begin(), KERNELHISTOGRAM_COMP_SPV.end()}
        , _kernelBlellochScan{KERNELBLELLOCHSCAN_COMP_SPV.begin(), KERNELBLELLOCHSCAN_COMP_SPV.end()}
        , _kernelBlockSum{KERNELBLOCKSUM_COMP_SPV.begin(), KERNELBLOCKSUM_COMP_SPV.end()}
        , _kernelIdCells{KERNELIDCELLS_COMP_SPV.begin(), KERNELIDCELLS_COMP_SPV.end()}
    {}


    template <typename FloatType>
    std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> ImplVulkan<FloatType>::simulate(const std::vector<Particle<FloatType>> &particles) {
        std::vector<float> positionsHost(particles.size() * 4, 0.0);
        std::vector<float> velocitiesHost(particles.size() * 4, 0.0);
        std::vector<float> forcesHost(particles.size() * 4, 0.0);
        std::vector<float> oldForcesHost(particles.size() * 4, 0.0);

        for (size_t p = 0; p < particles.size(); ++p) {
            positionsHost[p*4] = particles[p].getPosition()[0];
            positionsHost[p*4 + 1] = particles[p].getPosition()[1];
            positionsHost[p*4 + 2] = particles[p].getPosition()[2];

            velocitiesHost[p*4] = particles[p].getVelocity()[0];
            velocitiesHost[p*4 + 1] = particles[p].getVelocity()[1];
            velocitiesHost[p*4 + 2] = particles[p].getVelocity()[2];

            forcesHost[p*4] = particles[p].getForce()[0];
            forcesHost[p*4 + 1] = particles[p].getForce()[1];
            forcesHost[p*4 + 2] = particles[p].getForce()[2];
        }

        std::array<float, 3> boxMin = _config.boxMin;
        std::array<float, 3> boxMax = _config.boxMax;
        std::array<float, 3> boxSize = { boxMax[0] - boxMin[0], boxMax[1] - boxMin[1], boxMax[2] - boxMin[2] };
        std::array<int, 3> cellCounts = { 
            util::ceilDiv<uint>(boxSize[0], _config.h), 
            util::ceilDiv<uint>(boxSize[1], _config.h), 
            util::ceilDiv<uint>(boxSize[2], _config.h) };
        uint nCells = cellCounts[0] * cellCounts[1] * cellCounts[2];
        uint TILE_SIZE = _config.TILE_SIZE;
        uint nBlocks = (nCells + TILE_SIZE - 1) / TILE_SIZE;
        uint cellsLength = nBlocks * TILE_SIZE;

        std::vector<uint> cellsHost(cellsLength, 0);
        std::vector<int> particleIdxHost(particles.size() * 2, 0);
        std::vector<uint> idCellsHost(particles.size(), 0);

        auto positions = _manager.tensor(positionsHost);
        auto velocities = _manager.tensor(velocitiesHost);
        auto forces = _manager.tensor(forcesHost);
        auto oldForces = _manager.tensor(oldForcesHost);

        auto cells = _manager.tensor(cellsHost);
        auto particleIdx = _manager.tensor(particleIdxHost);
        auto idCells = _manager.tensor(idCellsHost);

        std::vector<std::shared_ptr<kp::Tensor>> params = {positions, velocities, forces, oldForces, cells, particleIdx, idCells};
        _sequence->template record<kp::OpTensorSyncDevice>(params)->eval();
        _timings.reset();

        for (int i = 0; i < _config.numberTimeSteps; ++i) {
            calculateHistogram({positions, cells, particleIdx}, cellCounts, boxMin, boxSize);
            exclusiveScanBlelloch(cells, cellsLength);
            calculateIdCells({particleIdx, idCells, cells});

            updatePositionsAndResetForce({positions, velocities, forces, oldForces});
            computeForces({positions, forces, particleIdx, cells, idCells}, cellCounts);
            updateVelocities({velocities, forces, oldForces});
            _sequence->template record<kp::OpTensorSyncLocal>(params)->eval();
        }

        printBuffer(positions, true);
        
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
    void ImplVulkan<FloatType>::calculateHistogram(const std::vector<std::shared_ptr<kp::Tensor>> &params, std::array<int, 3> cellCounts, std::array<float, 3> boxMin, std::array<float, 3> boxSize) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        struct PushHist {
            uint32_t numParticles;
            int32_t cCx;
            int32_t cCy;
            int32_t cCz;
            float bMinx;
            float bMiny;
            float bMinz;
            float bSizex;
            float bSizey;
            float bSizez;
        };

        PushHist pc{
            static_cast<uint32_t>(_config.size),
            cellCounts[0],
            cellCounts[1],
            cellCounts[2],
            boxMin[0],
            boxMin[1],
            boxMin[2],
            boxSize[0],
            boxSize[1],
            boxSize[2]
        };

        std::vector<uint32_t> pushData((sizeof(PushHist) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushHist));

        auto algorithm = _manager.algorithm(params, _kernelHistogram, workgroup, {}, pushData);

        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData);
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
        _sequence->template record<kp::OpAlgoDispatch>(algorithmBlelloch, pushData);

        // calculate prefix sum of block sum
        if (nBlocks > 1) {
            exclusiveScanBlelloch(blockSum, blockSumSize);

            // add block offset
            std::vector<uint32_t> pushData{totalLength};

            auto algorithmBlock = _manager.algorithm({data, blockSum}, _kernelBlockSum, workgroup, {}, pushData);

            // dispatch shader
            _sequence->template record<kp::OpAlgoDispatch>(algorithmBlock, pushData);
        }
    }


    template <typename FloatType>
    void ImplVulkan<FloatType>::calculateIdCells(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};
        std::vector<uint32_t> pushData{_config.size};

        auto algorithm = _manager.algorithm(params, _kernelIdCells, workgroup, {}, pushData);

        // dispatch shader
        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData);
    }


    template <typename FloatType>
    std::vector<uint> ImplVulkan<FloatType>::exclusiveScanNaive(std::vector<uint> vec) {
        std::vector<uint> result(vec.size(), 0);
        for (uint i = 0; i < vec.size() - 1; i++) {
            result[i + 1] = result[i] + vec[i];
        }
        return result;
    }


    template <typename FloatType>
    void ImplVulkan<FloatType>::benchmarkExclusiveScan(uint length, std::vector<uint> vec, const std::shared_ptr<kp::Tensor> &tensor) {
        std::cout << "number of elements: " << length << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        exclusiveScanBlelloch(tensor, length);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed1 = end - start;

        std::cout << "Blelloch exclusive scan took " << elapsed1.count() << " ms\n";

        start = std::chrono::high_resolution_clock::now();
        vec = exclusiveScanNaive(vec);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end - start;

        std::cout << "Naive exclusive scan took " << elapsed2.count() << " ms\n";

        std::cout << "Speedup factor: " << elapsed2.count() / elapsed1.count() << std::endl;
        std::cout << "Validation: " << validateExclusiveScan(vec, tensor) << std::endl;
    }


    template <typename FloatType>
    bool ImplVulkan<FloatType>::validateExclusiveScan(std::vector<uint> cpu, const std::shared_ptr<kp::Tensor> &gpuBuffer) {
        _sequence->record<kp::OpTensorSyncLocal>({ gpuBuffer });
        auto gpu = gpuBuffer->vector<uint>();

        bool matching = true;
        for (uint i = 0; i < cpu.size(); i++) {
            matching = matching && (cpu[i] == gpu[i]);
            if (!matching) {
                std::cout << i << ": " << cpu[i] << " != " << gpu[i] << std::endl;
                break;
            }
        }
        return matching; 
    }


    template <typename FloatType>
    void ImplVulkan<FloatType>::updatePositionsAndResetForce(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        struct PushPos {
            float globalForce_x;
            float globalForce_y;
            float globalForce_z;
            float dt;
            uint32_t numParticles;
        };

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

        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData);

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.positionUpdateForceResetTime += elapsed_nanoseconds;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::updateVelocities(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        struct PushVel {
            float dt;
            uint32_t numParticles;
        };

        PushVel pc{
            _config.deltaT,
            static_cast<uint32_t>(_config.size)
        };

        std::vector<uint32_t> pushData((sizeof(PushVel) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushVel));

        auto algorithm = _manager.algorithm(params, _kernelVelocity, workgroup, {}, pushData);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData);

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.velocityUpdateTime += elapsed_nanoseconds;
    }

    template <typename FloatType>
    void ImplVulkan<FloatType>::computeForces(const std::vector<std::shared_ptr<kp::Tensor>> &params, std::array<int, 3> cellCounts) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};
        
        struct PushFor {
            uint32_t numParticles;
            int32_t cCx;
            int32_t cCy;
            int32_t cCz;
        };

        PushFor pc{
            static_cast<uint32_t>(_config.size),
            cellCounts[0],
            cellCounts[1],
            cellCounts[2]
        };

        std::vector<uint32_t> pushData((sizeof(PushFor) + 3) / 4);
        std::memcpy(pushData.data(), &pc, sizeof(PushFor));

        auto algorithm = _manager.algorithm(params, _kernelForce, workgroup, {}, pushData);

        const auto start = std::chrono::high_resolution_clock::now();

        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushData);

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        _timings.forceUpdateTime += elapsed_nanoseconds;
    }

    template class ImplVulkan<float>;

} // namespace ppb
