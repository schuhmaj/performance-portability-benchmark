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
    class ImplSlangVulkan {

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
        std::vector<uint32_t> _kernelResetCells;

    public:

        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplSlangVulkan(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Runs the simulation for the configured total simulation time, performing position, force,
         * and velocity updates for each step using the velocity Verlet scheme.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::vector<Particle<FloatType>> Final state of all particles after the simulation.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

    private:
        /**
         * Syncs a specified buffer from the GPU and prints it. 
         * The passed bool allows for interpretation of floats and uints. 'true' corresponds to floats 
         * and 'false' corresponds to uints.
         * This function is a useful debugging tool.
         */
        void printBuffer(const std::shared_ptr<kp::Tensor> &buffer, bool floats);

        /**
         * Creates a histogram of the number of particles in each cell. This is used to generate a flat 
         * list of particle IDs with offsets corresponding to the cell. 
         */
        void calculateHistogram(const std::vector<std::shared_ptr<kp::Tensor>> &params, std::array<int, 3> cellCounts, std::array<float, 3> boxMin, std::array<float, 3> boxSize);

        /**
         * Calculates an exclusive scan of the buffer 'data' with length 'totalLength'.
         * Uses the Blelloch algorithm which was recommended here: 
         * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda 
         */
        void exclusiveScanBlelloch(const std::shared_ptr<kp::Tensor> &data, const uint totalLength);

        /**
         * Generates a flat buffer where particle IDs are sorted by their cell. This cell is used to 
         * find all particles in a cell.
         */
        void calculateIdCells(const std::vector<std::shared_ptr<kp::Tensor>> &params);

        /**
         * Resets the cell buffer to 0s so that the histogram can be rebuilt again using the calculateHistogram function.
         */
        void resetCells(const std::shared_ptr<kp::Tensor> &cells, uint nBlocks, uint cellsLength);

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