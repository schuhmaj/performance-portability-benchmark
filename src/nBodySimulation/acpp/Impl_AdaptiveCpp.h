/**
 * @file Impl_AdaptiveCpp.h
 *
 * Implements a classical n-body simulation using the Lennard-Jones potential and a simple velocity Verlet integrator.
 * Provides methods for updating particle positions, velocities, and computing inter-particle forces according
 * to the given simulation configuration. Designed to be used as a backend for benchmarking or analysis.
 */

#pragma once

#include <sycl/sycl.hpp>

#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"

#include "Parameters.h"
#include "ParticleContainer.h"

namespace ppb {

    /**
     * @class ImplAdaptiveCpp
     * Template for a classical n-body simulation using the Lennard-Jones potential and velocity Verlet integration.
     *
     * @tparam FloatType Floating-point type for simulation (float or double).
     */
    template<typename FloatType, AlgorithmType Algorithm = ppb::Naive<>>
    class ImplAdaptiveCpp {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;

        /**
         * Stores tje timings for position, velocity, and force updates
         */
        ParticleSimulationTimings _timings;

        /**
         * Manages the tasks on a device
         */
        sycl::queue _queue;

        /**
         * Particle Data in SoA form on device
         */
        sycl::vec<FloatType, 4> *_positions = nullptr;
        sycl::vec<FloatType, 4> *_velocities = nullptr;
        sycl::vec<FloatType, 4> *_forces = nullptr;
        sycl::vec<FloatType, 4> *_oldForces = nullptr;

        size_t _size = 0;

    public:

        /**
         * Type alias for the floating-point type used in this simulation.
         */
        using float_type = FloatType;

        /**
         * Constructs the simulation implementation.
         * @param config Simulation configuration with all necessary simulation parameters.
         */
        explicit ImplAdaptiveCpp(const ParticleSimulationConfig<FloatType> &config) : _config{config}, _timings{}, _queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}}, _size{_config.size} {}

        /**
         * Runs the simulation for the configured total simulation time, performing position, force,
         * and velocity updates for each step using the velocity Verlet scheme.
         *
         * @param particles Initial vector of particles to simulate (input is not modified).
         * @return std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> Final state of all particles after the simulation, paired with execution timings
         */
        std::pair<std::vector<Particle<FloatType>>, ParticleSimulationTimings> simulate(const std::vector<Particle<FloatType>> &particles) {
            // a copy of the passed particles. N * sizeof(Particle<FloatType>) overhead
            // can theoretically be freed once data is moved to SoA
            std::vector<Particle<FloatType>> particlesCopy = particles;

            // a container for particles in SoA form. N * 16 * sizeof(FloatType) overhead
            // build SoA from AoO
            ParticleContainer<FloatType, Algorithm> particle_container(particles, _config);

            sycl::queue q(sycl::default_selector{});
            int32_t* device_int = sycl::malloc_shared<int32_t>(1, q);
            if (device_int == nullptr) {
                std::cerr << "Shared malloc failed\n";
                std::exit(1);
            }
            device_int[0] = 42;
            q.wait();

            // copy of particles in SoA form in USM. N * 16 * sizeof(FloatType) overhead, on device
            // set up memory on device
            _positions  = static_cast<sycl::vec<FloatType, 4> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 4>), _size * sizeof(sycl::vec<FloatType, 4>), _queue));
            _velocities = static_cast<sycl::vec<FloatType, 4> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 4>), _size * sizeof(sycl::vec<FloatType, 4>), _queue));
            _forces     = static_cast<sycl::vec<FloatType, 4> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 4>), _size * sizeof(sycl::vec<FloatType, 4>), _queue));
            _oldForces  = static_cast<sycl::vec<FloatType, 4> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 4>), _size * sizeof(sycl::vec<FloatType, 4>), _queue));

            if (!_positions || !_velocities || !_forces || !_oldForces)  {
                throw std::runtime_error("USM allocation failed");
            }

            int32_t *cellsUSM = nullptr;
            int32_t *cell_countsUSM = nullptr;
            int32_t *overflowUSM = nullptr;
            if constexpr (Algorithm::kind == AlgorithmKinds::CellList) {
                // Spatial Subdivision in cells. N * 5 * sizeof(int32_t) overhead, on device
                cellsUSM = static_cast<int32_t *>(sycl::aligned_alloc_device(4 * sizeof(int32_t), 4 * ((particle_container.getCellsSize() - 1) / 4 + 1) * sizeof(int32_t), _queue));
                cell_countsUSM = static_cast<int32_t *>(sycl::aligned_alloc_device(4 * sizeof(int32_t), 4 * ((particle_container.getTotalCells() - 1) / 4 + 1) * sizeof(int32_t), _queue));
                overflowUSM = static_cast<int32_t *>(sycl::aligned_alloc_device(4 * sizeof(int32_t), 4 * ((_size - 1) / 4 + 1) * sizeof(int32_t), _queue));
                if (!cellsUSM || !cell_countsUSM || !overflowUSM) {
                    throw std::runtime_error("USM allocation failed");
                }

                particle_container.setCells(cellsUSM, cell_countsUSM, overflowUSM);
            }

            _queue.wait();

            // move data to device
            _queue.memcpy(_positions,  particle_container.getPositions(),  _size * sizeof(sycl::vec<FloatType, 4>));
            _queue.memcpy(_velocities, particle_container.getVelocities(), _size * sizeof(sycl::vec<FloatType, 4>));
            _queue.memcpy(_forces,     particle_container.getForces(),     _size * sizeof(sycl::vec<FloatType, 4>));
            _queue.memcpy(_oldForces,  particle_container.getOldForces(),  _size * sizeof(sycl::vec<FloatType, 4>));

            for (int i = 0; i < _config.numberTimeSteps; ++i) {
                updatePositionsAndResetForce();
                if constexpr (Algorithm::kind == AlgorithmKinds::Naive) {
                    computeForces_naive_atomic();
                } else if constexpr (Algorithm::kind == AlgorithmKinds::CellList) {
                    particle_container.buildCells(_queue, _positions);
                    computeForces_cell_based_atomic(&particle_container);
                }
                updateVelocities();
            }

            // move data to host
            _queue.memcpy(particle_container.getPositions(),    _positions,   _size * sizeof(sycl::vec<FloatType, 4>));
            _queue.memcpy(particle_container.getVelocities(),   _velocities,  _size * sizeof(sycl::vec<FloatType, 4>));
            _queue.memcpy(particle_container.getForces(),       _forces,      _size * sizeof(sycl::vec<FloatType, 4>));
            _queue.memcpy(particle_container.getOldForces(),    _oldForces,   _size * sizeof(sycl::vec<FloatType, 4>));
            _queue.wait();

            // convert SoA to AoO
            particle_container.extractParticleData(particlesCopy);

            sycl::free(_positions, _queue);
            sycl::free(_velocities, _queue);
            sycl::free(_forces, _queue);
            sycl::free(_oldForces, _queue);

            if constexpr (Algorithm::kind == AlgorithmKinds::CellList) {
                sycl::free(cellsUSM, _queue);
                sycl::free(cell_countsUSM, _queue);
                sycl::free(overflowUSM, _queue);
            }

            return std::make_pair(particlesCopy, _timings);
        }

    private:

        /**
         * Updates positions of all particles using velocity Verlet integration and resets their forces
         * with the configured global force.
         */
        void updatePositionsAndResetForce() {
            using namespace ppb::util;

            const sycl::vec<FloatType, 4> globalForce = {_config.globalForce[0], _config.globalForce[1], _config.globalForce[2], FloatType(0.0)};
            const auto deltaT = _config.deltaT;

            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> idx) {
                const FloatType m = FloatType(1.0);

                _oldForces[idx] = _forces[idx];

                _forces[idx] = globalForce;

                sycl::vec<FloatType, 4> vfac = _velocities[idx] * deltaT;

                sycl::vec<FloatType, 4> ffac = _forces[idx] * (deltaT * deltaT / (2 * m));

                _positions[idx][0] += vfac[0] + ffac[0];
                _positions[idx][1] += vfac[1] + ffac[1];
                _positions[idx][2] += vfac[2] + ffac[2];
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.positionUpdateForceResetTime += elapsed_nanoseconds;
        }

        /**
         * Updates velocities of all particles using the forces computed before and after the integration step.
         */
        void updateVelocities() {
            using namespace ppb::util;

            const auto deltaT = _config.deltaT;

            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> idx) {
                const FloatType m = FloatType(1.0);

                // change in velocity = (force + oldForce) * deltaT / (2 * mass)
                _velocities[idx] += (_forces[idx] + _oldForces[idx]) * (deltaT / (2 * m));
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.velocityUpdateTime += elapsed_nanoseconds;
        }

        /**
         * Computes the inter-particle forces for all particles using the Lennard-Jones potential.
         * The Algorithm is a naive n^2 all-to-all loop
         */
        void computeForces_naive_atomic() {
            using namespace ppb::util;

            // tuned to best ratio
            constexpr size_t local_size_x = std::is_same_v<float_t, FloatType> ? Algorithm::_local_size_x_float : Algorithm::_local_size_x_double;
            constexpr size_t local_size_y = std::is_same_v<float_t, FloatType> ? 1024 / Algorithm::_local_size_x_float : 1024 / Algorithm::_local_size_x_double;

            const size_t global_size_x = ((_size + local_size_x - 1) / local_size_x) * local_size_x;
            const size_t global_size_y = ((_size + local_size_y - 1) / local_size_y) * local_size_y;

            sycl::range<2> global_range(global_size_x, global_size_y);
            sycl::range<2> local_range(local_size_x, local_size_y);
            sycl::nd_range<2> nd_range(global_range, local_range);

            auto event = _queue.parallel_for(nd_range, [=, *this](sycl::nd_item<2> item) {
                size_t i = item.get_global_id(0);
                size_t j = item.get_global_id(1);

                if (i == j || i >= _size || j >= _size) return;

                const FloatType sigma = FloatType(1.0);
                const FloatType sigmaSquared = sigma * sigma;
                const FloatType epsilon = FloatType(1.0);
                const FloatType epsilon24 = epsilon * FloatType(24);

                // distance = position_i - position_j
                std::array<FloatType, 3> dist{
                    _positions[i][0] - _positions[j][0],
                    _positions[i][1] - _positions[j][1],
                    _positions[i][2] - _positions[j][2]
                };

                const FloatType distSquared = dot(dist, dist);

                const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
                FloatType lj6 = sigmaSquared * inverseDistSquared;
                lj6 = lj6 * lj6 * lj6;
                const FloatType lj12 = lj6 * lj6;
                const FloatType lj12m6 = lj12 - lj6;
                const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;
                const std::array<FloatType, 3> f = dist * fac;

                // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
                sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi0(_forces[i][0]);
                sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi1(_forces[i][1]);
                sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi2(_forces[i][2]);
                atomic_fi0.fetch_add(f[0]);
                atomic_fi1.fetch_add(f[1]);
                atomic_fi2.fetch_add(f[2]);
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }

        /**
         * Computes the inter-particle forces for all particles of neighboring cells using the Lennard-Jones potential.
         * The Algorithm is a CellList based loop over the same and neighboring cells
         */
        void computeForces_cell_based_atomic(ParticleContainer<FloatType, Algorithm> *particle_container) {
            using namespace ppb::util;

            int32_t *cell_counts = particle_container->getCellCounts();
            int32_t *cells = particle_container->getCells();
            int32_t cell_size = particle_container->getCellsSize() / particle_container->getTotalCells();
            std::array<int32_t, 3> cell_count = particle_container->getCellCount();
            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> i) {
                int32_t cell_index = *reinterpret_cast<int32_t *>(&_positions[i][3]);
                if (cell_index == -1) {
                    return;
                }
                for (int32_t x = -1; x <= 1; ++x) {
                    for (int32_t y = -1; y <= 1; ++y) {
                        for (int32_t z = -1; z <= 1; ++z) {
                            int32_t cell_index_now = cell_index + x + y * cell_count[0] + z * cell_count[0] * cell_count[1];
                            // sanity check
                            if (cell_index_now >= cell_count[0] * cell_count[1] * cell_count[2]) continue;
                            for (size_t cell_it = 0; cell_it < cell_counts[cell_index_now]; ++cell_it) {
                                sycl::id<1> j = cells[cell_index_now * cell_size + cell_it];
                                if (i == j) continue;

                                const FloatType sigma = FloatType(1.0);
                                const FloatType sigmaSquared = sigma * sigma;
                                const FloatType epsilon = FloatType(1.0);
                                const FloatType epsilon24 = epsilon * FloatType(24);

                                // distance = position_i - position_j
                                std::array<FloatType, 3> dist{
                                    _positions[i][0] - _positions[j][0],
                                    _positions[i][1] - _positions[j][1],
                                    _positions[i][2] - _positions[j][2]
                                };

                                const FloatType distSquared = dot(dist, dist);

                                const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
                                FloatType lj6 = sigmaSquared * inverseDistSquared;
                                lj6 = lj6 * lj6 * lj6;
                                const FloatType lj12 = lj6 * lj6;
                                const FloatType lj12m6 = lj12 - lj6;
                                const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;
                                const std::array<FloatType, 3> f = dist * fac;

                                // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
                                sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi0(_forces[i][0]);
                                sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi1(_forces[i][1]);
                                sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_fi2(_forces[i][2]);
                                atomic_fi0.fetch_add(f[0]);
                                atomic_fi1.fetch_add(f[1]);
                                atomic_fi2.fetch_add(f[2]);
                            }
                        }
                    }
                }
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }

    };
} // namespace ppb