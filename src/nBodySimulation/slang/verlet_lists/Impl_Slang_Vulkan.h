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

        std::vector<uint32_t> _kernelCountNeighbors;
        std::vector<uint32_t> _kernelBlellochScan;
        std::vector<uint32_t> _kernelBlockSum;
        std::vector<uint32_t> _kernelVerlet;

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

        long retrieve_timestamps();

        /**
         * Syncs a specified buffer from the GPU and prints it. 
         * The passed bool allows for interpretation of floats and uints. 'true' corresponds to floats 
         * and 'false' corresponds to uints.
         * This function is a useful debugging tool.
         */
        void printBuffer(const std::shared_ptr<kp::Tensor> &buffer, bool floats);

        /**
         * This function is called for each particle and iterates over all other particles to count
         * how many can be considered neighbors using the 'influenceRadius' hyperparameter in the config.
         * This function is the first step in creating a flattened verlet list, and is also needed for 
         * determining how much memory needs to be allocated to the verlet lists.
         */
        void countNeighbors(const std::vector<std::shared_ptr<kp::Tensor>> &params);

        /**
         * Calculates an exclusive scan of the buffer 'data' with length 'totalLength'.
         * Uses the Blelloch algorithm which was recommended here: 
         * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda 
         */
        void exclusiveScanBlelloch(const std::shared_ptr<kp::Tensor> &data, const uint totalLength);

        /**
         * Writes the flattened verlet list. The shader is called for each particle and iterates over each other 
         * particle again to write the particle ID to the correct buffer address.
         */
        std::shared_ptr<kp::Tensor> createVerletList(const std::shared_ptr<kp::Tensor> &positions, const std::shared_ptr<kp::Tensor> &neighborsStarts);
        
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
        void computeForces(const std::vector<std::shared_ptr<kp::Tensor>> &params);


    };
} // namespace ppb