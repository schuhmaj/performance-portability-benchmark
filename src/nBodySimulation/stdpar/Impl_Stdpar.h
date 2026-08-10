/**
 * @file Impl_Stdpar.h
 *
 * Implements the classical n-body simulation (Lennard-Jones potential, velocity Verlet integrator)
 * using the ISO C++ standard parallel algorithms (<execution> with std::execution::par_unseq).
 * When compiled with a stdpar-offloading toolchain (NVHPC -stdpar=gpu, ROCm --hipstdpar,
 * Intel -fsycl-pstl-offload=gpu, or AdaptiveCpp --acpp-stdpar), the parallel loops execute on the GPU
 * while the std::vector storage is transparently placed in unified/managed memory.
 */

#pragma once
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include <chrono>
#include <optional>
#include <vector>

namespace ppb {

    /**
     * SoA (Structure of Arrays) representation of the particles. In contrast to the OpenMP variant,
     * plain std::vector storage suffices: the stdpar toolchains interpose the heap allocations with
     * unified/managed memory, so no explicit host/device copies are required.
     */
    template <typename FloatType>
    struct StdparParticleSoA {

        /**
         * Creates the SoA representation from the given particles.
         * @param ref the particles used to fill positions, velocities and forces (old forces are zeroed)
         */
        explicit StdparParticleSoA(const std::vector<Particle<FloatType>> &ref);

        /**
         * Converts the SoA representation back to a vector of particles (based on the reference
         * particles given at construction).
         * @return the particles with updated positions, velocities and forces
         */
        std::vector<Particle<FloatType>> toParticles() const;

        const std::vector<Particle<FloatType>> &_ref;

        std::vector<FloatType> positions;
        std::vector<FloatType> velocities;
        std::vector<FloatType> forces;
        std::vector<FloatType> oldForces;

        /**
         * Materialized counting range [0, particle count) used to drive the per-particle kernels.
         * A lazy std::views::iota would express the same thing, but its iterators are not portable
         * across the stdpar backends (oneDPL cannot map them to device memory and their
         * difference_type is __int128, which SPIR-V targets do not support), so the indices are held
         * in ordinary unified-memory storage like every other array here.
         */
        std::vector<size_t> indices;
    };

    /**
     * @class ImplStdpar
     * Template for a classical n-body simulation using the Lennard-Jones potential and velocity Verlet
     * integration, parallelized with the ISO C++ standard parallel algorithms.
     *
     * @tparam FloatType Floating-point type for simulation (float or double).
     */
    template<typename FloatType>
    class ImplStdpar {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

        /**
         * The SoA representation of the particles (constructed per simulate() call).
         */
        std::optional<StdparParticleSoA<FloatType>> _particles;

    public:

        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplStdpar(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Runs the simulation for the configured total simulation time, performing position, force,
         * and velocity updates for each step using the velocity Verlet scheme.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return Final state of all particles after the simulation and the accumulated timings.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

    private:

        /**
         * Updates positions of all particles using velocity Verlet integration and resets their forces
         * with the configured global force.
         */
        void updatePositionsAndResetForce();

        /**
         * Updates velocities of all particles using the forces computed before and after the integration step.
         */
        void updateVelocities();

        /**
         * Computes the inter-particle forces for all particles using the Lennard-Jones potential.
         */
        void computeForces();

    };
} // namespace ppb
