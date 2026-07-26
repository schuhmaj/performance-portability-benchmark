/**
 * @file Impl_Raja.h
 *
 * Implements an n-body simulation using RAJA for portability across various parallel architectures.
 * Utilizes the Lennard-Jones potential and a velocity Verlet integrator for the motion and force calculations.
 * This implementation enables high-performance execution on CPUs and GPUs by leveraging the RAJA abstraction layer.
 * Designed to be used as a backend for benchmarking and performance comparison.
 *
 * Note: This header intentionally does not include any RAJA headers, so that it can be consumed from
 * translation units which are not compiled by the device compiler. All RAJA usage lives in Impl_Raja.cpp.
 */

#pragma once

#include <optional>
#include <vector>

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {

    /**
     * @struct RajaParticleSoA
     * Structure of Arrays (SoA) container for particle attributes optimized for RAJA parallelism.
     *
     * Stores particle positions, velocities, forces, and related data in device buffers (raw pointers
     * allocated through the RAJA/camp resource of the active backend) for efficient bulk memory access
     * and parallel computation on CPUs or GPUs. The data is laid out in a structure-of-arrays style
     * with shape [N][3] flattened to N*3 row-major.
     *
     * @tparam FloatType Floating-point type for numeric data (e.g., float or double).
     */
    template <typename FloatType>
    struct RajaParticleSoA {
        /**
         * Device buffer of particle positions, shape [N][3] (row-major).
         */
        FloatType *positions{nullptr};

        /**
         * Device buffer of particle velocities, shape [N][3] (row-major).
         */
        FloatType *velocities{nullptr};

        /**
         * Device buffer of particle forces, shape [N][3] (row-major).
         */
        FloatType *forces{nullptr};

        /**
         * Device buffer of previous forces for velocity Verlet integration, shape [N][3] (row-major).
         */
        FloatType *oldForces{nullptr};

        /**
         * Reference to the original vector of particles (used as a data source during initialization and conversion).
         */
        const std::vector<Particle<FloatType>> &_ref;

        /**
         * Constructs a RajaParticleSoA from a standard vector of Particle objects.
         * Allocates the device buffers and copies the input particle data into the SoA structure.
         *
         * @param particles Input vector of Particle<FloatType> objects to initialize from.
         */
        explicit RajaParticleSoA(const std::vector<Particle<FloatType>> &particles);

        /**
         * Frees the device buffers.
         */
        ~RajaParticleSoA();

        RajaParticleSoA(const RajaParticleSoA &) = delete;
        RajaParticleSoA &operator=(const RajaParticleSoA &) = delete;

        /**
         * Copies the current SoA data back into a standard vector of Particle objects.
         * This involves copying data from device to host and assembling Particle instances.
         *
         * @return A vector of Particle<FloatType> reflecting the state stored in the device buffers.
         */
        std::vector<Particle<FloatType>> toParticles();

        /**
         * Returns the number of particles managed by this structure.
         *
         * @return Particle count (size of the arrays).
         */
        size_t size() const;
    };

    /**
     * @class ImplRaja
     * Templated n-body simulation using RAJA for parallelism, the Lennard-Jones potential, and velocity Verlet
     * integration.
     *
     * @tparam FloatType Floating-point type for simulation (e.g., float or double).
     */
    template <typename FloatType>
    class ImplRaja {
    protected:

        /**
         * Simulation configuration which holds parameters such as particle count, global forces, simulation time, etc.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * The SoA GPU structure. It is initialized each time the simulate() functions is called
         */
        std::optional<RajaParticleSoA<FloatType>> _particles{std::nullopt};

        /**
         * Stores the timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

    public:
        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplRaja(const ParticleSimulationConfig<FloatType> &config);

        /**
         * Destroys the simulation implementation.
         *
         * Note: The body is written out instead of using '= default' on purpose. hipcc/ clang infer
         * '__host__ __device__' for defaulted special member functions, and since a virtual destructor is
         * referenced by the vtable, the device compilation pass would then emit it for the GPU as well.
         * Its body destroys the std::optional<RajaParticleSoA> member, whose destructor calls into the
         * host-only camp resource (hipFree, hipStreamCreate, ...), which fails at device link time with
         * undefined symbols. A user-provided destructor is plain '__host__', so the device vtable slot
         * stays empty and nothing host-only is dragged into the device image.
         */
        virtual ~ImplRaja() {}

        /**
         * Runs the simulation for the configured total time using parallel RAJA kernels to update
         * positions, velocities, and compute forces at each step.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return Final state of all particles after the simulation together with the accumulated timings.
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles);

        /**
         * Updates positions of all particles on the device using the velocity Verlet integrator,
         * and resets each particle's force to the configured global force in parallel.
         */
        void updatePositionsAndResetForce();

        /**
         * Updates velocities of all particles on the device based on forces before and after the integration
         * step, using parallel execution.
         */
        void updateVelocities();

        /**
         * Computes the inter-particle forces using the Lennard-Jones potential for all particles on the device,
         * accumulating the results in parallel.
         */
        virtual void computeForces();
    };
} // namespace ppb
