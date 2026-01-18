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
     * @tparam Algorithm used Algorithm as defined in "Parameters.h"
     */
    template<typename FloatType, AlgorithmType Algorithm = ppb::Naive<>>
    class ImplAdaptiveCpp {

        /**
         * Simulation configuration with parameters such as particle count, force, time step, and bounds.
         */
        ParticleSimulationConfig<FloatType> _config;
        static constexpr FloatType cutoff = Algorithm::_cutoff;

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
        sycl::vec<FloatType, 3> *_positions = nullptr;
        sycl::vec<FloatType, 3> *_velocities = nullptr;
        sycl::vec<FloatType, 3> *_forces = nullptr;
        sycl::vec<FloatType, 3> *_oldForces = nullptr;

        size_t _size = 0;

        /**
         * Records any permutation of indices, so they can be reversed
         */
        uint32_t *_indices = nullptr;

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
            // a working copy of the passed particles. N * sizeof(Particle<FloatType>) overhead
            std::vector<Particle<FloatType>> particlesCopy = particles;

            for (size_t i = _size; i < pad(_size); ++i) {
                particlesCopy.push_back(Particle{std::array<FloatType, 3>{__builtin_inf(), __builtin_inf(), __builtin_inf()}});
            }

            // a container for particles in SoA form. N * 16 * sizeof(FloatType) overhead
            // build SoA from AoO
            ParticleContainer<FloatType, Algorithm> particle_container(particlesCopy, _config, cutoff, _queue);

            // copy of particles in SoA form in USM. N * 16 * sizeof(FloatType) overhead, on device
            // set up memory for data on device
            _positions  = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            _velocities = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            _forces     = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            _oldForces  = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));

            if (!_positions || !_velocities || !_forces || !_oldForces)  {
                throw std::runtime_error("Device allocation of particles failed");
            }

            // if sorting is enabled, permutation of indices is recorded. N * sizeof(uint32_t) overhead
            if constexpr (Algorithm::_sorter != SorterKinds::None) {
                _indices = static_cast<uint32_t *>(sycl::aligned_alloc_device(4 * sizeof(uint32_t), pad(_size) * sizeof(uint32_t), _queue));
                if (!_indices) {
                    throw std::runtime_error("Device allocation of indices failed");
                }
                _queue.parallel_for(sycl::range<1>(pad(_size)), [=, *this](sycl::id<1> i) {
                    _indices[i] = static_cast<uint32_t>(i);
                });
            }

            _queue.wait();

            // move data to device
            _queue.memcpy(_positions,  particle_container.getPositions(),  pad(_size) * sizeof(sycl::vec<FloatType, 3>));
            _queue.memcpy(_velocities, particle_container.getVelocities(), pad(_size) * sizeof(sycl::vec<FloatType, 3>));
            _queue.memcpy(_forces,     particle_container.getForces(),     pad(_size) * sizeof(sycl::vec<FloatType, 3>));
            _queue.memcpy(_oldForces,  particle_container.getOldForces(),  pad(_size) * sizeof(sycl::vec<FloatType, 3>));
            _queue.wait();

            // main simulation logic
            for (int i = 0; i < _config.numberTimeSteps; ++i) {
                updatePositionsAndResetForce();

                // optional sorting of data in memory
                if constexpr (Algorithm::_sorter == SorterKinds::MergeProjection) {
                    if (i % sorter_frequency == 0) {
                        merge_projection_sort();
                        _queue.wait();
                    }
                }
                if constexpr (Algorithm::_sorter == SorterKinds::MergeCellID) {
                    if (i % sorter_frequency == 0) {
                        merge_cellid_sort(particle_container);
                        _queue.wait();
                    }
                }

                // selection of algorithm for force computation
                if constexpr (Algorithm::kind == AlgorithmKinds::Naive && Algorithm::_sorter == SorterKinds::MergeProjection) {
                    computeForces_naive_sorted();
                } else if constexpr (Algorithm::kind == AlgorithmKinds::CellList && Algorithm::_sorter == SorterKinds::MergeCellID) {
                    if (i % sorter_frequency == 0) {
                        build_cells_merge_sort(particle_container);
                    }
                    computeForces_cellList_sorted(particle_container);
                } else if constexpr (Algorithm::kind == AlgorithmKinds::CellList) {
                    if (i % sorter_frequency == 0) {
                        build_cells_merge_sort(particle_container);
                    }
                    computeForces_cellList_unsorted(particle_container);
                } else {
                    computeForces_naive_unsorted();
                }

                updateVelocities();
            }

            // undo any permutation to the particle array
            if constexpr (Algorithm::_sorter != SorterKinds::None) {
                reverse_sort();
                _queue.wait();
                sycl::free(_indices, _queue);
            }

            // move data to host
            _queue.memcpy(particle_container.getPositions(),    _positions,   _size * sizeof(sycl::vec<FloatType, 3>));
            _queue.memcpy(particle_container.getVelocities(),   _velocities,  _size * sizeof(sycl::vec<FloatType, 3>));
            _queue.memcpy(particle_container.getForces(),       _forces,      _size * sizeof(sycl::vec<FloatType, 3>));
            _queue.memcpy(particle_container.getOldForces(),    _oldForces,   _size * sizeof(sycl::vec<FloatType, 3>));
            _queue.wait();

            // convert SoA to AoO
            particle_container.extractParticleData(particlesCopy);

            sycl::free(_positions, _queue);
            sycl::free(_velocities, _queue);
            sycl::free(_forces, _queue);
            sycl::free(_oldForces, _queue);

            // undo particle padding
            for (size_t i = _size; i < pad(_size); ++i) {
                particlesCopy.pop_back();
            }

            return std::make_pair(particlesCopy, _timings);
        }

    private:
        /**
         * swaps the data of two particles. _indices keeps track
         * @param i first position
         * @param j second position
         */
        void pswap(size_t i, size_t j) const {
            std::swap(_positions[i], _positions[j]);
            std::swap(_velocities[i], _velocities[j]);
            std::swap(_forces[i], _forces[j]);
            std::swap(_oldForces[i], _oldForces[j]);
            std::swap(_indices[i], _indices[j]);
        }

        /**
         * Sorts all particles of the simulation based on their (1,1,1)-projection using warp sort.
         * Small blocks are first sorted using bitonic sort. Resulting sequences are sorted using merge sort.
         * Because we have the same number of stages over the same number of small blocks, there is no measurable performance advantage over mergesort.
         * Testing shows this working correctly on its own, but there is a bug when used as a replacement of merge_projection_sort()
         */
        void warp_projection_sort() {
            for (int stage = 2; stage < BLOCK_SIZE; stage <<= 1) {
                for (int stride = stage >> 1; stride > 0; stride >>= 1) {
                    _queue.parallel_for(sycl::nd_range<1>(pad(_size), BLOCK_SIZE / 2), [=, *this](const sycl::nd_item<1> &item) {
                        // bitwise math allows the the 1..size range to map to a range without duplicate comparisons.
                        // a naive implementation would launch comparators for both A?B and B?A and then cancel B?A
                        size_t i = item.get_group(0) * BLOCK_SIZE + (((item.get_local_id(0) & ~(stride - 1)) << 1) | (item.get_local_id(0) & (stride - 1)));
                        size_t j = i ^ stride;
                        FloatType i_projection = _positions[i][0] + _positions[i][1] + _positions[i][2];
                        FloatType j_projection = _positions[j][0] + _positions[j][1] + _positions[j][2];
                        // swap according to direction and evaluation of comparator
                        if ((i & stage) == 0 && i_projection > j_projection || (i & stage) != 0 && i_projection < j_projection) {
                            pswap(i, j);
                        }
                    }).wait();
                }
            }
            for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
                _queue.parallel_for(sycl::nd_range<1>(pad(_size), BLOCK_SIZE / 2), [=, *this](const sycl::nd_item<1> &item) {
                    // bitwise math allows the the 1..size range to map to a range without duplicate comparisons.
                    // a naive implementation would launch comparators for both A?B and B?A and then cancel B?A
                    size_t i = item.get_group(0) * BLOCK_SIZE + (((item.get_local_id(0) & ~(stride - 1)) << 1) | (item.get_local_id(0) & (stride - 1)));
                    size_t j = i ^ stride;
                    FloatType i_projection = _positions[i][0] + _positions[i][1] + _positions[i][2];
                    FloatType j_projection = _positions[j][0] + _positions[j][1] + _positions[j][2];
                    if (i_projection > j_projection) {
                        pswap(i, j);
                    }
                }).wait();
            }
            for (size_t stage = BLOCK_SIZE; stage < std::bit_ceil(pad(_size)); stage <<= 1) {
                for (size_t stride = stage; stride > 0; stride >>= 1) {
                    size_t offset_base = stride & (stage - 1);
                    _queue.parallel_for(sycl::range<1>((pad(_size) + 1) / 2), [=, *this](const sycl::id<1> idx) {
                        // bitwise math allows the the 1..size range to map to a range without duplicate comparisons.
                        // a naive implementation would launch comparators for both A?B and B?A and then cancel B?A
                        size_t combined_offset = idx.get(0);
                        size_t block_offset = offset_base + ((combined_offset << 1) & ~((stride << 1) - 1));
                        size_t inner_offset = combined_offset & (stride - 1);
                        // check if comparator is valid, not every stage needs every value compared
                        if ((block_offset + inner_offset) / (stage * 2) == (block_offset + inner_offset + stride) / (stage * 2) && (block_offset + inner_offset + stride) < pad(_size)) {
                            size_t i = block_offset + inner_offset;
                            size_t j = block_offset + inner_offset + stride;
                            FloatType i_projection = _positions[i][0] + _positions[i][1] + _positions[i][2];
                            FloatType j_projection = _positions[j][0] + _positions[j][1] + _positions[j][2];
                            if (i_projection > j_projection) {
                                pswap(i, j);
                            }
                        }
                    }).wait();
                }
            }
        }

        /**
         * Sorts all particles of the simulation based on their (1,1,1)-projection using merge sort.
         * This sorting algorithm is implemented to equal a sorting network like batcher's odd-even mergesort
         * https://en.wikipedia.org/wiki/Batcher_odd-even_mergesort
         */
        void merge_projection_sort() {
            for (size_t stage = 1; stage < std::bit_ceil(pad(_size)); stage <<= 1) {
                for (size_t stride = stage; stride > 0; stride >>= 1) {
                    size_t offset_base = stride & (stage - 1);
                    auto event = _queue.parallel_for(sycl::range<1>((pad(_size) + 1) / 2), [=, *this](const sycl::id<1> idx) {
                        // bitwise math allows the the 1..size range to map to a range without duplicate comparisons.
                        // a naive implementation would launch comparators for both A?B and B?A and then cancel B?A
                        size_t combined_offset = idx.get(0);
                        size_t block_offset = offset_base + ((combined_offset << 1) & ~((stride << 1) - 1));
                        size_t inner_offset = combined_offset & (stride - 1);
                        // check if comparator is valid, not every stage needs every value compared
                        if ((block_offset + inner_offset) / (stage * 2) == (block_offset + inner_offset + stride) / (stage * 2) && (block_offset + inner_offset + stride) < pad(_size)) {
                            size_t i = block_offset + inner_offset;
                            size_t j = block_offset + inner_offset + stride;
                            FloatType i_projection = _positions[i][0] + _positions[i][1] + _positions[i][2];
                            FloatType j_projection = _positions[j][0] + _positions[j][1] + _positions[j][2];
                            if (i_projection > j_projection) {
                                pswap(i, j);
                            }
                        }
                    });
                    _queue.wait();
                    auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
                    auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
                    const double elapsed_nanoseconds = end - start;
                    _timings.sortingTime += elapsed_nanoseconds;
                }
            }
        }

        /**
        * Sorts all particles of the simulation based on their cellid using merge sort.
         * This sorting algorithm is implemented to equal a sorting network like batcher's odd-even mergesort
         * https://en.wikipedia.org/wiki/Batcher_odd-even_mergesort
         */
        void merge_cellid_sort(ParticleContainer<FloatType, Algorithm> &particle_container) {
            uint32_t *keys = particle_container.getKeys();
            uint32_t *values = particle_container.getValues();
            if (!keys || !values) {
                build_kv(particle_container);
                keys = particle_container.getKeys();
                values = particle_container.getValues();
            }
            for (size_t stage = 1; stage < std::bit_ceil(pad(_size)); stage <<= 1) {
                for (size_t stride = stage; stride > 0; stride >>= 1) {
                    size_t offset_base = stride & (stage - 1);
                    auto event = _queue.parallel_for(sycl::range<1>((pad(_size) + 1) / 2), [=, *this](const sycl::id<1> idx) {
                        // bitwise math allows the the 1..size range to map to a range without duplicate comparisons.
                        // a naive implementation would launch comparators for both A?B and B?A and then cancel B?A
                        size_t combined_offset = idx.get(0);
                        size_t block_offset = offset_base + ((combined_offset << 1) & ~((stride << 1) - 1));
                        size_t inner_offset = combined_offset & (stride - 1);
                        // check if comparator is valid, not every stage needs every value compared
                        if ((block_offset + inner_offset) / (stage * 2) == (block_offset + inner_offset + stride) / (stage * 2) && (block_offset + inner_offset + stride) < pad(_size)) {
                            size_t i = block_offset + inner_offset;
                            size_t j = block_offset + inner_offset + stride;
                            FloatType i_cellid = keys[i];
                            FloatType j_cellid = keys[j];
                            if (i_cellid > j_cellid) {
                                pswap(i, j);
                                std::swap(keys[i], keys[j]);
                                std::swap(values[i], values[j]);
                            }
                        }
                    });
                    _queue.wait();
                    auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
                    auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
                    const double elapsed_nanoseconds = end - start;
                    _timings.sortingTime += elapsed_nanoseconds;
                }
            }
        }

        /**
         * Builds the keys and values array to be used in Cell Lists
         */
        void build_kv(ParticleContainer<FloatType, Algorithm> &particle_container) {
            // get keys and values, create them if not present
            uint32_t *keys = particle_container.getKeys();
            uint32_t *values = particle_container.getValues();
            if (!keys || !values) {
                keys = static_cast<uint32_t *>(sycl::aligned_alloc_device(4 * alignof(uint32_t), pad(_size) * sizeof(uint32_t), _queue));
                values = static_cast<uint32_t *>(sycl::aligned_alloc_device(4 * alignof(uint32_t), pad(_size) * sizeof(uint32_t), _queue));
                _queue.wait();
                if (!keys || !values) {
                    throw std::runtime_error("Device allocation of kv arrays failed");
                }
                particle_container.setKeys(keys);
                particle_container.setValues(values);
            }

            // fill keys and values with cell_ids and indices
            const std::array<size_t, 3> cell_count = {particle_container.getCellCount()[0], particle_container.getCellCount()[1], particle_container.getCellCount()[2]};
            std::array<FloatType, 3> boxMin = {_config.boxMin[0], _config.boxMin[1], _config.boxMin[2]};
            std::array<FloatType, 3> boxMax = {_config.boxMax[0], _config.boxMax[1], _config.boxMax[2]};
            auto event = _queue.parallel_for<class compute_cell_id>(sycl::range<1>(pad(_size)), [=, *this](sycl::id<1> i) {
                if (_positions[i][0] > boxMin[0] && _positions[i][1] > boxMin[1] && _positions[i][2] > boxMin[2]
                    && _positions[i][0] < boxMax[0] && _positions[i][1] < boxMax[1] && _positions[i][2] < boxMax[2]) {
                    const std::array<uint32_t, 3> indices {
                        static_cast<uint32_t>(sycl::floor((_positions[i][0] - boxMin[0]) / ((boxMax[0] - boxMin[0]) / (cell_count[0] - 2)))) + 1,
                        static_cast<uint32_t>(sycl::floor((_positions[i][1] - boxMin[1]) / ((boxMax[1] - boxMin[1]) / (cell_count[1] - 2)))) + 1,
                        static_cast<uint32_t>(sycl::floor((_positions[i][2] - boxMin[2]) / ((boxMax[2] - boxMin[2]) / (cell_count[2] - 2)))) + 1
                    };
                    keys[i] = indices[0] + indices[1] * cell_count[0] + indices[2] * cell_count[0] * cell_count[1];
                } else {
                    keys[i] = std::numeric_limits<uint32_t>::max();
                }
                values[i] = i;
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            double elapsed_nanoseconds = end - start;
            _timings.structuralTime += elapsed_nanoseconds;
        }

        /**
         * Sorts all particles of the simulation based on their cell_id using bitonic sort.
        * Sorters are launched for the smallest power of two greater than _size and out-of-bounds accesses are skipped.
         * This sorting algorithm is implemented to equal a sorting network like batcher's odd-even mergesort
         * https://en.wikipedia.org/wiki/Batcher_odd-even_mergesort
         */
        void build_cells_merge_sort(ParticleContainer<FloatType, Algorithm> &particle_container) {
            // get keys and values and create and build them if not present
            uint32_t *keys = particle_container.getKeys();
            uint32_t *values = particle_container.getValues();
            uint32_t *cells = particle_container.getCells();
            if (!keys || !values) {
                build_kv(particle_container);
                keys = particle_container.getKeys();
                values = particle_container.getValues();
            }
            if (!cells) {
                cells = static_cast<uint32_t *>(sycl::aligned_alloc_device(4 * alignof(uint32_t), pad(particle_container.getTotalCells()) * sizeof(uint32_t), _queue));
                _queue.wait();
                if (!cells) {
                    throw std::runtime_error("Device allocation of cells array failed");
                }
                particle_container.setCells(cells);
            }

            // sort both keys and values by keys
            if (Algorithm::_sorter != SorterKinds::MergeCellID) {
                for (size_t stage = 1; stage < std::bit_ceil(pad(_size)); stage <<= 1) {
                    for (size_t stride = stage; stride > 0; stride >>= 1) {
                        size_t offset_base = stride & (stage - 1);
                        auto event = _queue.parallel_for(sycl::range<1>((pad(_size) + 1) / 2), [=, *this](const sycl::id<1> idx) {
                            // bitwise math allows the the 1..size range to map to a range without duplicate comparisons.
                            // a naive implementation would launch comparators for both A?B and B?A and then cancel B?A
                            size_t combined_offset = idx.get(0);
                            size_t block_offset = offset_base + ((combined_offset << 1) & ~((stride << 1) - 1));
                            size_t inner_offset = combined_offset & (stride - 1);
                            // check if comparator is valid, not every stage needs every value compared
                            if ((block_offset + inner_offset) / (stage * 2) == (block_offset + inner_offset + stride) / (stage * 2) && (block_offset + inner_offset + stride) < pad(_size)) {
                                size_t i = block_offset + inner_offset;
                                size_t j = block_offset + inner_offset + stride;
                                FloatType i_cell = keys[i];
                                FloatType j_cell = keys[j];
                                if (i_cell > j_cell) {
                                    std::swap(keys[i], keys[j]);
                                    std::swap(values[i], values[j]);
                                }
                            }
                        });
                        _queue.wait();
                        auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
                        auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
                        const double elapsed_nanoseconds = end - start;
                        _timings.structuralTime += elapsed_nanoseconds;
                    }
                }
            }

            // compute the cell offset on the host. This can in theory be done on the device,
            // but is an inherently linear operation that only benefits from parallelism at very large problem sizes and uniform density distributions
            uint32_t *host_keys = sycl::aligned_alloc_host<uint32_t>(4 * alignof(uint32_t), pad(_size) * sizeof(uint32_t), _queue);
            uint32_t *host_cells = sycl::aligned_alloc_host<uint32_t>(4 * alignof(uint32_t), pad(particle_container.getTotalCells()) * sizeof(uint32_t), _queue);
            _queue.wait();
            if (!host_keys || !host_cells) {
                throw std::runtime_error("Host allocation of cells failed");
            }
            _queue.memcpy(host_keys, keys, pad(_size) * sizeof(uint32_t)).wait();
            size_t current_key = 0;
            for (size_t i = 0; i < _size; ++i) {
                // cells are offset to point to the first matching cell_id in the list.
                // this allows for (cell[k], cell[k+1]) to be any size from 0 to problem size and to span the relevant particles (excluding cell[k+1])
                if (host_keys[i] != current_key) {
                    for (size_t j = current_key + 1; j <= host_keys[i]; ++j) {
                        host_cells[j] = i;
                    }
                    current_key = host_keys[i];
                }
            }
            for (size_t i = current_key + 1; i < particle_container.getTotalCells(); ++i) {
                host_cells[i] = _size;
            }

            // move the computed cell offsets to device memory
            _queue.memcpy(cells, host_cells, pad(particle_container.getTotalCells()) * sizeof(uint32_t)).wait();

            sycl::free(host_cells, _queue);
            sycl::free(host_keys, _queue);
        }

        /**
         * Restores particle order to the beginning of the simulation
         * temporarily allocates 16 * sizeof(FloatType) * pad(_size) Bytes of memory
         */
        void reverse_sort() {
            // allocate space in memory, so that data does not need to be swapped in-place, requiring explicit synchronization or atomics
            sycl::vec<FloatType, 3> *unsorted_positions  = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            sycl::vec<FloatType, 3> *unsorted_velocities = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            sycl::vec<FloatType, 3> *unsorted_forces     = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            sycl::vec<FloatType, 3> *unsorted_oldForces  = static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_device(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            _queue.wait();

            if (!_positions || !_velocities || !_forces || !_oldForces)  {
                throw std::runtime_error("Device allocation of particles (for reversing sorts) failed");
            }

            // write data from the sorted array to the unsorted array based on permutations
            _queue.parallel_for(sycl::range<1>(pad(_size)), [=, *this](sycl::id<1> i) {
                unsorted_positions[_indices[i]] = _positions[i];
                unsorted_velocities[_indices[i]] = _velocities[i];
                unsorted_forces[_indices[i]] = _forces[i];
                unsorted_oldForces[_indices[i]] = _oldForces[i];
            }).wait();

            sycl::free(_positions, _queue);
            sycl::free(_velocities, _queue);
            sycl::free(_forces, _queue);
            sycl::free(_oldForces, _queue);
            _queue.wait();

            // swap pointers of sorted data and unsorted data, effectively reversing any sorting
            _positions = unsorted_positions;
            _velocities = unsorted_velocities;
            _forces = unsorted_forces;
            _oldForces = unsorted_oldForces;
        }

        /**
         * Updates positions of all particles using velocity Verlet integration and resets their forces
         * with the configured global force.
         */
        void updatePositionsAndResetForce() {
            using namespace ppb::util;

            const sycl::vec<FloatType, 3> globalForce = {_config.globalForce[0], _config.globalForce[1], _config.globalForce[2]};
            const auto deltaT = _config.deltaT;

            std::swap(_oldForces, _forces);

            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> idx) {
                const FloatType m = FloatType(1.0);

                _forces[idx] = globalForce;

                // change in position = (velocity * deltaT) + (force * deltaT^2 / (2 * mass))
                _positions[idx] += _velocities[idx] * deltaT + _forces[idx] * (deltaT * deltaT / (2 * m));
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
         * The Algorithm is a naive n^2 all-to-all loop, parallelized on a per-particle basis
         */
        void computeForces_naive_unsorted() {
            using namespace ppb::util;

            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> idx) {
                size_t i = idx[0];
                // temporary variable to accumulate forces. the final result can be used to write to memory once
                sycl::vec<FloatType, 3> fi{0,0,0};

                const FloatType sigma = FloatType(1.0);
                const FloatType sigmaSquared = sigma * sigma;
                const FloatType epsilon = FloatType(1.0);
                const FloatType epsilon24 = epsilon * FloatType(24);

                // all-to-all iteration
                for (size_t j = 0; j < _size; ++j) {
                    if (i == j || j >= _size)
                        continue;

                    // distance = position_i - position_j
                    sycl::vec<FloatType, 3> dist = _positions[i] - _positions[j];

                    const FloatType distSquared = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];

                    if (distSquared > cutoff * cutoff) {
                        continue;
                    }

                    const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
                    FloatType lj6 = sigmaSquared * inverseDistSquared;
                    lj6 = lj6 * lj6 * lj6;
                    const FloatType lj12 = lj6 * lj6;
                    const FloatType lj12m6 = lj12 - lj6;
                    const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;

                    // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
                    fi += dist * fac;
                }
                _forces[i] += fi;
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }

        /**
         * Computes the inter-particle forces for all particles using the Lennard-Jones potential.
         * The Algorithm is a naive n^2 all-to-all loop, parallelized on a per-particle basis
         */
        void computeForces_naive_sorted() {
            using namespace ppb::util;

            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> idx) {
                size_t i = idx[0];
                // temporary variable to accumulate forces. the final result can be used to write to memory once
                sycl::vec<FloatType, 3> fi{0,0,0};

                const FloatType sigma = FloatType(1.0);
                const FloatType sigmaSquared = sigma * sigma;
                const FloatType epsilon = FloatType(1.0);
                const FloatType epsilon24 = epsilon * FloatType(24);

                // forward iteration until j is too far from i
                for (size_t j = i + 1; j < _size; ++j) {
                    // distance = position_i - position_j
                    sycl::vec<FloatType, 3> dist = _positions[i] - _positions[j];

                    if (dist[0] + dist[1] + dist[2] < -cutoff) {
                        break;
                    }

                    const FloatType distSquared = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];

                    if (distSquared > cutoff * cutoff) {
                        continue;
                    }

                    const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
                    FloatType lj6 = sigmaSquared * inverseDistSquared;
                    lj6 = lj6 * lj6 * lj6;
                    const FloatType lj12 = lj6 * lj6;
                    const FloatType lj12m6 = lj12 - lj6;
                    const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;

                    // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
                    fi += dist * fac;
                }
                // backwards iteration until j is too far from i
                for (ssize_t j = static_cast<ssize_t>(i) - 1; j >= 0; --j) {
                    // distance = position_i - position_j
                    sycl::vec<FloatType, 3> dist = _positions[i] - _positions[j];

                    if (dist[0] + dist[1] + dist[2] > cutoff) {
                        break;
                    }

                    const FloatType distSquared = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];

                    if (distSquared > cutoff * cutoff) {
                        continue;
                    }

                    const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
                    FloatType lj6 = sigmaSquared * inverseDistSquared;
                    lj6 = lj6 * lj6 * lj6;
                    const FloatType lj12 = lj6 * lj6;
                    const FloatType lj12m6 = lj12 - lj6;
                    const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;

                    // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
                    fi += dist * fac;
                }
                _forces[i] += fi;
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }

        /**
         * Computes the inter-particle forces for all particles of neighboring cells using the Lennard-Jones potential.
         * The Algorithm is a CellList based loop over the same and neighboring cells, parallelized on a per-particle basis.
         */
        void computeForces_cellList_unsorted(ParticleContainer<FloatType, Algorithm> &particle_container) {
            using namespace ppb::util;

            uint32_t *keys = particle_container.getKeys();
            uint32_t *values = particle_container.getValues();
            uint32_t *cells = particle_container.getCells();

            std::array<size_t, 3> cell_count = particle_container.getCellCount();
            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> idx) {
                // cell_index points to the cell of particle i
                uint32_t cell_index = keys[idx[0]];
                // i is the true data index of particle i
                uint32_t i = values[idx[0]];
                // temporary variable to accumulate forces. the final result can be used to write to memory once
                sycl::vec<FloatType, 3> fi{0,0,0};
                // neighbor iteration offsets
                for (int32_t x = -1; x <= 1; ++x) {
                    for (int32_t y = -1; y <= 1; ++y) {
                        for (int32_t z = -1; z <= 1; ++z) {
                            uint32_t cell_index_now = cell_index + x + y * cell_count[0] + z * cell_count[0] * cell_count[1];
                            // cell_it points to the starting index of the cells particles in keys and values
                            for (size_t cell_it = cells[cell_index_now]; cell_it < cells[cell_index_now + 1] && cell_it < _size; ++cell_it) {
                                uint32_t j = values[cell_it];
                                if (i == j) continue;

                                const FloatType sigma = FloatType(1.0);
                                const FloatType sigmaSquared = sigma * sigma;
                                const FloatType epsilon = FloatType(1.0);
                                const FloatType epsilon24 = epsilon * FloatType(24);

                                // distance = position_i - position_j
                                sycl::vec<FloatType, 3> dist = _positions[i] - _positions[j];

                                const FloatType distSquared = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];

                                if (distSquared > cutoff * cutoff) {
                                    continue;
                                }

                                const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
                                FloatType lj6 = sigmaSquared * inverseDistSquared;
                                lj6 = lj6 * lj6 * lj6;
                                const FloatType lj12 = lj6 * lj6;
                                const FloatType lj12m6 = lj12 - lj6;
                                const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;

                                // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
                                fi += dist * fac;
                            }
                        }
                    }
                }
                _forces[i] += fi;
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }

        /**
         * Computes the inter-particle forces for all particles of neighboring cells using the Lennard-Jones potential.
         * The Algorithm is a CellList based loop over the same and neighboring cells, parallelized on a per-particle basis.
         */
        void computeForces_cellList_sorted(ParticleContainer<FloatType, Algorithm> &particle_container) {
            using namespace ppb::util;

            uint32_t *keys = particle_container.getKeys();
            uint32_t *values = particle_container.getValues();
            uint32_t *cells = particle_container.getCells();

            std::array<size_t, 3> cell_count = particle_container.getCellCount();
            auto event = _queue.parallel_for(sycl::range<1>(_size), [=, *this](sycl::id<1> idx) {
                // cell_index points to the cell of particle i
                uint32_t cell_index = keys[idx[0]];
                // i is the true data index of particle i
                uint32_t i = idx[0];
                // temporary variable to accumulate forces. the final result can be used to write to memory once
                sycl::vec<FloatType, 3> fi{0,0,0};
                // neighbor iteration offsets
                for (int32_t x = -1; x <= 1; ++x) {
                    for (int32_t y = -1; y <= 1; ++y) {
                        for (int32_t z = -1; z <= 1; ++z) {
                            uint32_t cell_index_now = cell_index + x + y * cell_count[0] + z * cell_count[0] * cell_count[1];
                            // cell_it points to the starting index of the cells particles in keys and values
                            for (size_t cell_it = cells[cell_index_now]; cell_it < cells[cell_index_now + 1] && cell_it < _size; ++cell_it) {
                                uint32_t j = cell_it;
                                if (i == j) continue;

                                const FloatType sigma = FloatType(1.0);
                                const FloatType sigmaSquared = sigma * sigma;
                                const FloatType epsilon = FloatType(1.0);
                                const FloatType epsilon24 = epsilon * FloatType(24);

                                // distance = position_i - position_j
                                sycl::vec<FloatType, 3> dist = _positions[i] - _positions[j];

                                const FloatType distSquared = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];

                                if (distSquared > cutoff * cutoff) {
                                    continue;
                                }

                                const FloatType inverseDistSquared = FloatType(1.0) / distSquared;
                                FloatType lj6 = sigmaSquared * inverseDistSquared;
                                lj6 = lj6 * lj6 * lj6;
                                const FloatType lj12 = lj6 * lj6;
                                const FloatType lj12m6 = lj12 - lj6;
                                const FloatType fac = epsilon24 * (lj12 + lj12m6) * inverseDistSquared;

                                // change in force = dist * (epsilon * 24 * (2 * lj12 - lj6) * inverse distance square)
                                fi += dist * fac;
                            }
                        }
                    }
                }
                _forces[i] += fi;
            });
            _queue.wait();
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            const double elapsed_nanoseconds = end - start;
            _timings.forceUpdateTime += elapsed_nanoseconds;
        }
    };
} // namespace ppb