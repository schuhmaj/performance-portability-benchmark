/**
 * @file ParticleContainer.h
 *
 * Manages a set of Particles according to different Algorithms.
 * Provides methods for setting and retrieving the set of Particles and Particle information, as well as updating the Container instance
 */

#pragma once

#include <sycl/sycl.hpp>

#include "nBodySimulation/Particle.h"

#include "Parameters.h"

size_t make_multiple(size_t original, size_t base) {
    return ((original - 1) / base + 1) * base;
}

namespace ppb {

    template <typename FloatType, AlgorithmType Algorithm = ppb::Naive<>>
    class ParticleContainer {
        sycl::vec<FloatType, 4> *positions;
        sycl::vec<FloatType, 4> *velocities;
        sycl::vec<FloatType, 4> *forces;
        sycl::vec<FloatType, 4> *oldForces;

        const ParticleSimulationConfig<FloatType> _config;

static constexpr size_t MEMORYSIZE = 64 * sizeof(FloatType);

        int32_t *cells;
        int32_t *cell_counts;
        int32_t *overflow;

        size_t cells_size;
        std::array<int32_t, 3> cell_count;
        size_t total_cells;
        size_t overflow_count;

        size_t _size;

    public:

        // build SoA from AoO
        explicit ParticleContainer(const std::vector<Particle<FloatType>> &particles, const ParticleSimulationConfig<FloatType> _config) : _config(_config), _size{_config.size} {
            positions = static_cast<sycl::vec<FloatType, 4> *>(std::aligned_alloc(alignof(sycl::vec<FloatType, 4>), make_multiple(sizeof(sycl::vec<FloatType, 4>) * _size, MEMORYSIZE)));
            velocities = static_cast<sycl::vec<FloatType, 4> *>(std::aligned_alloc(alignof(sycl::vec<FloatType, 4>), make_multiple(sizeof(sycl::vec<FloatType, 4>) * _size, MEMORYSIZE)));
            forces = static_cast<sycl::vec<FloatType, 4> *>(std::aligned_alloc(alignof(sycl::vec<FloatType, 4>), make_multiple(sizeof(sycl::vec<FloatType, 4>) * _size, MEMORYSIZE)));
            oldForces = static_cast<sycl::vec<FloatType, 4> *>(std::aligned_alloc(alignof(sycl::vec<FloatType, 4>), make_multiple(sizeof(sycl::vec<FloatType, 4>) * _size, MEMORYSIZE)));

            for (size_t i = 0; i < _size; ++i) {
                positions[i] = {particles[i].getPosition()[0], particles[i].getPosition()[1], particles[i].getPosition()[2], FloatType(0)};
                velocities[i] = {particles[i].getVelocity()[0], particles[i].getVelocity()[1], particles[i].getVelocity()[2], FloatType(0)};
                forces[i] = {particles[i].getForce()[0], particles[i].getForce()[1], particles[i].getForce()[2], FloatType(0)};
                oldForces[i] = {particles[i].getOldForce()[0], particles[i].getOldForce()[1], particles[i].getOldForce()[2], FloatType(0)};
            }

            int32_t MAX_CELL_COUNT = 1;
            if constexpr (Algorithm::kind == AlgorithmKinds::CellList) {
                // MAX_CELL_COUNT is computed dynamically based on number of particles
                // this improves the algorithm in situations where the bounding box is large, but there's not many particles
                MAX_CELL_COUNT = static_cast<int32_t>(std::floor(std::cbrt(static_cast<double>(_size) / Algorithm::_particles_per_cell)));

                cell_count = {
                    std::max(1, std::min(MAX_CELL_COUNT, static_cast<int32_t>(std::floor((_config.boxMax[0] - _config.boxMin[0]) / FloatType(3.5))))) + 2,
                    std::max(1, std::min(MAX_CELL_COUNT, static_cast<int32_t>(std::floor((_config.boxMax[1] - _config.boxMin[1]) / FloatType(3.5))))) + 2,
                    std::max(1, std::min(MAX_CELL_COUNT, static_cast<int32_t>(std::floor((_config.boxMax[2] - _config.boxMin[2]) / FloatType(3.5))))) + 2
                };
                total_cells = cell_count[0] * cell_count[1] * cell_count[2];
                // perfectly even distribution would mean that every cell has about the same avg_occupancy.
                // DENSITY_FACTOR allows cells to contain DENSITY_FACTOR times as many particles as expected from
                // avg_occupancy. any particles that still overflow any given cell are stored in overflow and will be
                // treated separately.
                const int32_t avg_occupancy = (_size - 1) / ((cell_count[0] - 2) * (cell_count[1] - 2) * (cell_count[2] - 2)) + 1;
                cells_size = Algorithm::_max_cell_density * avg_occupancy * total_cells;
            }
        }

        ~ParticleContainer() {
            std::free(positions);
            std::free(velocities);
            std::free(forces);
            std::free(oldForces);
        }

        // convert SoA to AoO
        void extractParticleData(std::vector<Particle<FloatType>> &particles) {
            for (size_t i = 0; i < _size; ++i) {
                particles[i].setPosition({positions[i][0], positions[i][1], positions[i][2]});
                particles[i].setVelocity({velocities[i][0], velocities[i][1], velocities[i][2]});
                particles[i].setForce({forces[i][0], forces[i][1], forces[i][2]});
                particles[i].setOldForce({oldForces[i][0], oldForces[i][1], oldForces[i][2]});
            }
        }

        void buildCells(sycl::queue &queue, sycl::vec<FloatType, 4> *positions) {
            queue.memset(cell_counts, 0, 4 * ((total_cells - 1) / 4 + 1) * sizeof(int32_t)).wait();
            std::array<FloatType, 3> boxSize {
                _config.boxMax[0] - _config.boxMin[0],
                _config.boxMax[1] - _config.boxMin[1],
                _config.boxMax[2] - _config.boxMin[2]
            };
            std::array<FloatType, 3> boxMin {
                _config.boxMin[0],
                _config.boxMin[1],
                _config.boxMin[2]
            };
            std::array<int32_t, 3> cell_count_tick {
                cell_count[0],
                cell_count[1],
                cell_count[2]
            };
            size_t total_cells_tick = total_cells;
            size_t cells_size_tick = cells_size;
            int32_t *cells_tick = cells;
            int32_t *cell_counts_tick = cell_counts;
            int32_t *overflow_tick = overflow;
            size_t *overflow_count_tick = &overflow_count;
            queue.parallel_for(sycl::range<1>(_size), [=](sycl::id<1> idx) {
                // clamping factor 0.999999 allows for boxMax to fit into the cell box.
                // TODO: make boxMax exclusive in future iterations?
                std::array<int32_t, 3> indices{
                    static_cast<int32_t>(sycl::floor(0.999999 * (positions[idx][0] - boxMin[0]) / (boxSize[0] / (cell_count_tick[0] - 2)))) + 1,
                    static_cast<int32_t>(sycl::floor(0.999999 * (positions[idx][1] - boxMin[1]) / (boxSize[1] / (cell_count_tick[1] - 2)))) + 1,
                    static_cast<int32_t>(sycl::floor(0.999999 * (positions[idx][2] - boxMin[2]) / (boxSize[2] / (cell_count_tick[2] - 2)))) + 1
                };
                int32_t flat_index = indices[0] + indices[1] * cell_count_tick[0] + indices[2] * cell_count_tick[0] * cell_count_tick[1];
                if (indices[0] < 1 || indices[1] < 1 || indices[2] < 1 || indices[0] >= cell_count_tick[0] - 1 || indices[1] >= cell_count_tick[1] - 1 || indices[2] >= cell_count_tick[2] - 1) {
                    flat_index = -1;
                } else if (flat_index < total_cells_tick && cell_counts_tick[flat_index] < cells_size_tick / total_cells_tick) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_cell_counts(cell_counts_tick[flat_index]);
                    int32_t cell_counts_now = atomic_cell_counts.fetch_add(1);
                    cells_tick[flat_index * (cells_size_tick / total_cells_tick) + cell_counts_now] = static_cast<int32_t>(idx);
                } else {
                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_overflow_count(*overflow_count_tick);
                    size_t overflow_count_now = atomic_overflow_count.fetch_add(static_cast<size_t>(1));
                    overflow_tick[overflow_count_now] = static_cast<int32_t>(idx);
                }
                int32_t *cell_keeper = reinterpret_cast<int32_t *>(&positions[idx][3]);
                *cell_keeper = flat_index;
            }).wait();
        }

        sycl::vec<FloatType, 4> *getPositions() { return positions; }
        sycl::vec<FloatType, 4> *getVelocities() { return velocities; }
        sycl::vec<FloatType, 4> *getForces() { return forces; }
        sycl::vec<FloatType, 4> *getOldForces() { return oldForces; }

        int32_t *getCells() const { return cells; }
        int32_t *getCellCounts() const { return cell_counts; }
        int32_t *getOverflow() const { return overflow; }

        void setCells(int32_t *cellsUSM, int32_t *cell_countsUSM, int32_t *overflowUSM) {
            cells = cellsUSM;
            cell_counts = cell_countsUSM;
            overflow = overflowUSM;
        }

        size_t &getCellsSize() { return cells_size; }
        std::array<int32_t, 3> &getCellCount() { return cell_count;}
        size_t &getTotalCells() { return total_cells; }
        size_t &getOverflowCount() { return overflow_count; }
    };
}