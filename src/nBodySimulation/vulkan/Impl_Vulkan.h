#pragma once
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"
#include "common/vulkan/VulkanUtility.h"
#include <iostream>
#include <optional>
#include <chrono>

namespace ppb {

    template<typename FloatType>
    class ImplVulkan {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

        vulkan_utility::VulkanManager _manager;

        std::shared_ptr<kp::Sequence> _sequence;

        std::vector<uint32_t> _kernelPosition;
        std::vector<uint32_t> _kernelVelocity;
        std::vector<uint32_t> _kernelForce;

        std::vector<uint32_t> _kernelHistogram;
        std::vector<uint32_t> _kernelBlellochScan;
        std::vector<uint32_t> _kernelBlockSum;
        std::vector<uint32_t> _kernelIdCells;

    public:

        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplVulkan(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Runs the simulation for the configured total simulation time, performing position, force,
         * and velocity updates for each step using the velocity Verlet scheme.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::vector<Particle<FloatType>> Final state of all particles after the simulation.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

    private:

        void printBuffer(const std::shared_ptr<kp::Tensor> &buffer, bool floats);

        void calculateHistogram(const std::vector<std::shared_ptr<kp::Tensor>> &params, std::array<int, 3> cellCounts, std::array<float, 3> boxMin, std::array<float, 3> boxSize);

        void exclusiveScanBlelloch(const std::shared_ptr<kp::Tensor> &data, const uint totalLength);

        void calculateIdCells(const std::vector<std::shared_ptr<kp::Tensor>> &params);

        std::vector<uint> exclusiveScanNaive(std::vector<uint> vec);

        void benchmarkExclusiveScan(uint length, std::vector<uint> vec, const std::shared_ptr<kp::Tensor> &tensor);

        bool validateExclusiveScan(std::vector<uint> cpu, const std::shared_ptr<kp::Tensor> &gpuBuffer);

        /**
         * Updates positions of all particles using velocity Verlet integration and resets their forces
         * with the configured global force.
         */
        void updatePositionsAndResetForce(const std::vector<std::shared_ptr<kp::Tensor>> &params);

        /**
         * Updates velocities of all particles using the forces computed before and after the integration step.
         */
        void updateVelocities(const std::vector<std::shared_ptr<kp::Tensor>> &params);

        /**
         * Computes the inter-particle forces for all particles using the Lennard-Jones potential.
         */
        void computeForces(const std::vector<std::shared_ptr<kp::Tensor>> &params, std::array<int, 3> cellCounts);

    };
} // namespace ppb