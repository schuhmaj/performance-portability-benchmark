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

namespace ppb {

    template <typename FloatType, AlgorithmType Algorithm = ppb::Naive<>>
    class ParticleContainer {
        /**
         * data fields for SoA construction
         */
        sycl::vec<FloatType, 3> *positions;
        sycl::vec<FloatType, 3> *velocities;
        sycl::vec<FloatType, 3> *forces;
        sycl::vec<FloatType, 3> *oldForces;

        /**
         * fields to build a dynamically sized cell list
         */
        uint32_t *keys = nullptr;
        uint32_t *values = nullptr;
        uint32_t *cells = nullptr;

        /**
         * cell list dynamic parameters
         */
        size_t cells_size;
        std::array<size_t, 3> cell_count;
        size_t total_cells;

        size_t _size;
        sycl::queue _queue;

    public:

        // build SoA from AoO
        explicit ParticleContainer(const std::vector<Particle<FloatType>> &particles, const ParticleSimulationConfig<FloatType> _config, const double cutoff, sycl::queue _queue) : _size{particles.size()}, _queue{_queue} {
            positions =     static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_host(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            velocities =    static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_host(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            forces =        static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_host(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            oldForces =     static_cast<sycl::vec<FloatType, 3> *>(sycl::aligned_alloc_host(alignof(sycl::vec<FloatType, 3>), pad(_size) * sizeof(sycl::vec<FloatType, 3>), _queue));
            _queue.wait();

            if (!positions || !velocities || !forces || !oldForces) {
                throw std::runtime_error("Host allocation of particles failed");
            }

            // build SoA from particle vector
            for (size_t i = 0; i < _size; ++i) {
                positions[i] = {particles[i].getPosition()[0], particles[i].getPosition()[1], particles[i].getPosition()[2]};
                velocities[i] = {particles[i].getVelocity()[0], particles[i].getVelocity()[1], particles[i].getVelocity()[2]};
                forces[i] = {particles[i].getForce()[0], particles[i].getForce()[1], particles[i].getForce()[2]};
                oldForces[i] = {particles[i].getOldForce()[0], particles[i].getOldForce()[1], particles[i].getOldForce()[2]};
            }

            if constexpr (Algorithm::kind == AlgorithmKinds::CellList) {
                // MAX_CELL_COUNT is computed dynamically based on number of particles
                // this improves the algorithm in situations where the bounding box is large, but there's not many particles
                size_t MAX_CELL_COUNT = static_cast<size_t>(std::floor(std::cbrt(static_cast<double>(_size) / Algorithm::_particles_per_cell)));

                cell_count = {
                    std::max<size_t>(1, std::min(MAX_CELL_COUNT, static_cast<size_t>(std::floor((_config.boxMax[0] - _config.boxMin[0]) / cutoff)))) + 2,
                    std::max<size_t>(1, std::min(MAX_CELL_COUNT, static_cast<size_t>(std::floor((_config.boxMax[1] - _config.boxMin[1]) / cutoff)))) + 2,
                    std::max<size_t>(1, std::min(MAX_CELL_COUNT, static_cast<size_t>(std::floor((_config.boxMax[2] - _config.boxMin[2]) / cutoff)))) + 2
                };
                total_cells = cell_count[0] * cell_count[1] * cell_count[2];
            }
        }

        ~ParticleContainer() {
            sycl::free(positions, _queue);
            sycl::free(velocities, _queue);
            sycl::free(forces, _queue);
            sycl::free(oldForces, _queue);

            if (Algorithm::kind == AlgorithmKinds::CellList) {
                sycl::free(keys, _queue);
                sycl::free(values, _queue);
                sycl::free(cells, _queue);
            }
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

        sycl::vec<FloatType, 3> *getPositions() { return positions; }
        void setPositions(sycl::vec<FloatType, 3> *positions_tmp) { sycl::free(positions, _queue); positions = positions_tmp; }
        sycl::vec<FloatType, 3> *getVelocities() { return velocities; }
        void setVelocities(sycl::vec<FloatType, 3> *velocities_tmp) { sycl::free(velocities, _queue); velocities = velocities_tmp; }
        sycl::vec<FloatType, 3> *getForces() { return forces; }
        void setForces(sycl::vec<FloatType, 3> *forces_tmp) { sycl::free(forces, _queue); forces = forces_tmp; }
        sycl::vec<FloatType, 3> *getOldForces() { return oldForces; }
        void setOldForces(sycl::vec<FloatType, 3> *oldForces_tmp) { sycl::free(oldForces, _queue); oldForces = oldForces_tmp; }

        size_t &getTotalCells() { return total_cells; }
        std::array<size_t, 3> &getCellCount() { return cell_count; }

        uint32_t *getKeys() { return keys; }
        void setKeys(uint32_t *keys_) { keys = keys_; }
        uint32_t *getValues() { return values;}
        void setValues(uint32_t *values_) { values = values_; }
        uint32_t *getCells() { return cells; }
        void setCells(uint32_t *cells_) { cells = cells_; }
    };
}