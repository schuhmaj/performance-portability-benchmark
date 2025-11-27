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

        std::array<float, 3> boxMin = _config.boxMin;
        std::array<float, 3> boxMax = _config.boxMax;
        std::array<float, 3> boxSize = { boxMax[0] - boxMin[0], boxMax[1] - boxMin[1], boxMax[2] - boxMin[2] };
        std::array<int, 3> cellCounts = { int(boxSize[0] / _config.h), int(boxSize[1] / _config.h), int(boxSize[2] / _config.h) };
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
    void ImplVulkan<FloatType>::calculateHistogram(const std::vector<std::shared_ptr<kp::Tensor>> &params, std::array<int, 3> cellCounts, std::array<float, 3> boxMin, std::array<float, 3> boxSize) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        std::vector<float> pushConstants{*reinterpret_cast<float*>(&_config.size), *reinterpret_cast<float*>(&cellCounts[0]), *reinterpret_cast<float*>(&cellCounts[1]), *reinterpret_cast<float*>(&cellCounts[2]), boxMin[0], boxMin[1], boxMin[2], boxSize[0], boxSize[1], boxSize[2]};

        auto algorithm = _manager.algorithm(params, _kernelHistogram, workgroup, {}, pushConstants);

        _sequence->template record<kp::OpAlgoDispatch>(algorithm ,pushConstants)->eval();
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
        std::vector<uint32_t> pushConstants{totalLength, TILE_SIZE};

        auto algorithmBlelloch = _manager.algorithm(
            {data, blockSum},
            _kernelBlellochScan,
            workgroup,
            {},
            pushConstants
        );

        // dispatch shader
        _sequence->template record<kp::OpAlgoDispatch>(algorithmBlelloch, pushConstants)->eval();

        // calculate prefix sum of block sum
        if (nBlocks > 1) {
            exclusiveScanBlelloch(blockSum, blockSumSize);

            // add block offset
            std::vector<uint32_t> pushConstants{totalLength};

            auto algorithmBlock = _manager.algorithm(
                {data, blockSum},
                _kernelBlockSum,
                workgroup,
                {},
                pushConstants
            );

            // dispatch shader
            _sequence->template record<kp::OpAlgoDispatch>(algorithmBlock, pushConstants)->eval();
        }
    }


    template <typename FloatType>
    void ImplVulkan<FloatType>::calculateIdCells(const std::vector<std::shared_ptr<kp::Tensor>> &params) {
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};
        std::vector<uint32_t> pushConstants{_config.size};

        auto algorithm = _manager.algorithm(
            params,
            _kernelIdCells,
            workgroup,
            {},
            pushConstants
        );

        // dispatch shader
        _sequence->template record<kp::OpAlgoDispatch>(algorithm, pushConstants)->eval();
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
        _sequence->record<kp::OpTensorSyncLocal>({ gpuBuffer })->eval();
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
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
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
        uint TILE_SIZE = _config.TILE_SIZE;
        const uint groups = util::ceilDiv<uint>(_config.size, TILE_SIZE);
        kp::Workgroup workgroup{{groups, 1, 1}};

        std::vector<uint> pushConstants{static_cast<uint>(_config.size)};

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


