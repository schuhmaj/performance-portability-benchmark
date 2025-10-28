#pragma once

#include <cstring>

#include <sycl/sycl.hpp>

#include "Particle.h"

namespace ppb {
    template <typename FloatType>
    class ParticleContainer {
        FloatType *positions;
        FloatType *velocities;
        FloatType *forces;
        FloatType *oldForces;

        const ParticleSimulationConfig<FloatType> _config;

#define DENSITY_FACTOR 4

        int32_t *cells;
        int32_t *cell_counts;
        int32_t *overflow;

        size_t cells_size;
        std::array<int32_t, 3> cell_count;
        size_t total_cells;
        size_t overflow_count;

        size_t size;

    public:

        // build SoA from AoO
        explicit ParticleContainer(const std::vector<Particle<FloatType>> &particles, const ParticleSimulationConfig<FloatType> _config) : _config(_config) {
            size = particles.size();

            positions = static_cast<FloatType *>(std::aligned_alloc(4 * sizeof(FloatType), 4 * sizeof(FloatType) * size));
            velocities = static_cast<FloatType *>(std::aligned_alloc(4 * sizeof(FloatType), 4 * sizeof(FloatType) * size));
            forces = static_cast<FloatType *>(std::aligned_alloc(4 * sizeof(FloatType), 4 * sizeof(FloatType) * size));
            oldForces = static_cast<FloatType *>(std::aligned_alloc(4 * sizeof(FloatType), 4 * sizeof(FloatType) * size));

            for (size_t i = 0; i < size; ++i) {
                std::copy_n(particles[i].getPosition().begin(), 3, positions + 4 * i);
                std::copy_n(particles[i].getVelocity().begin(), 3, velocities + 4 * i);
                std::copy_n(particles[i].getForce().begin(), 3, forces + 4 * i);
                std::copy_n(particles[i].getOldForce().begin(), 3, oldForces + 4 * i);
            }

            cell_count = {
                std::max(1, static_cast<int32_t>(std::floor((_config.boxMax[0] - _config.boxMin[0]) / FloatType(2.5)))),
                std::max(1, static_cast<int32_t>(std::floor((_config.boxMax[1] - _config.boxMin[1]) / FloatType(2.5)))),
                std::max(1, static_cast<int32_t>(std::floor((_config.boxMax[2] - _config.boxMin[2]) / FloatType(2.5)))),
            };
            total_cells = cell_count[0] * cell_count[1] * cell_count[2];
            // perfectly even distribution would mean that every cell has about the same avg_occupancy.
            // DENSITY_FACTOR allows cells to contain DENSITY_FACTOR times as many particles as expected from
            // avg_occupancy. any particles that still overflow any given cell are stored in overflow and will be
            // treated separately.
            const int32_t avg_occupancy = (size - 1) / total_cells + 1;
            cells_size = DENSITY_FACTOR * avg_occupancy * total_cells;
            overflow_count = 0;
        }

        ~ParticleContainer() {
            std::free(positions);
            std::free(velocities);
            std::free(forces);
            std::free(oldForces);
        }

        // convert SoA to AoO
        void extractParticleData(std::vector<Particle<FloatType>> &particles) {
            for (size_t i = 0; i < size; ++i) {
                particles[i].setPosition({positions[4 * i], positions[4 * i + 1], positions[4 * i + 2]});
                particles[i].setVelocity({velocities[4 * i], velocities[4 * i + 1], velocities[4 * i + 2]});
                particles[i].setForce({forces[4 * i], forces[4 * i + 1], forces[4 * i + 2]});
                particles[i].setOldForce({oldForces[4 * i], oldForces[4 * i + 1], oldForces[4 * i + 2]});
            }
        }

        void buildCells(sycl::queue &queue, FloatType *positionsUSM) {
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
            queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
                std::array<int32_t, 3> indices{
                    static_cast<int32_t>(sycl::floor((positionsUSM[4 * idx + 0] - boxMin[0]) / (boxSize[0] / cell_count_tick[0]))),
                    static_cast<int32_t>(sycl::floor((positionsUSM[4 * idx + 1] - boxMin[1]) / (boxSize[1] / cell_count_tick[1]))),
                    static_cast<int32_t>(sycl::floor((positionsUSM[4 * idx + 2] - boxMin[2]) / (boxSize[2] / cell_count_tick[2])))
                };
                int32_t flat_index = indices[0] + indices[1] * cell_count_tick[0] + indices[2] * cell_count_tick[0] * cell_count_tick[1];
                // temporary preset for position = boxMax edge case
                flat_index = 0;


                if (flat_index < total_cells_tick && cell_counts_tick[flat_index] < cells_size_tick / total_cells_tick) {
                    cells_tick[cell_counts_tick[flat_index]] = static_cast<int32_t>(idx);
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_cell_counts(cell_counts_tick[flat_index]);
                    atomic_cell_counts.fetch_add(1);
                } else {
                    overflow_tick[*overflow_count_tick] = static_cast<int32_t>(idx);
                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>atomic_overflow_count(*overflow_count_tick);
                    atomic_overflow_count.fetch_add(1);
                }
                int32_t *cell_keeper = reinterpret_cast<int32_t *>(&positionsUSM[4 * idx + 3]);
                *cell_keeper = flat_index;
            }).wait();
        }

        FloatType *getPositions() { return positions; }
        FloatType *getVelocities() { return velocities; }
        FloatType *getForces() { return forces; }
        FloatType *getOldForces() { return oldForces; }

        int32_t *getCells() const { return cells; }
        int32_t *getCellCounts() const { return cell_counts; }
        int32_t *getOverflow() { return overflow; }

        void setCells(int32_t *cellsUSM, int32_t *cell_countsUSM, int32_t *overflowUSM, sycl::queue &queue) {
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