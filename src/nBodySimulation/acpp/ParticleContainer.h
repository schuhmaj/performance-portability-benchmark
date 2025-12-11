#pragma once

#include "Particle.h"

namespace ppb {
    template <typename FloatType>
    class ParticleContainer {
        FloatType *positions;
        FloatType *velocities;
        FloatType *forces;
        FloatType *oldForces;

        size_t size;

    public:

        // build SoA from AoO
        explicit ParticleContainer(const std::vector<Particle<FloatType>> &particles) {
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

        FloatType *getPositions() { return positions; }
        FloatType *getVelocities() { return velocities; }
        FloatType *getForces() { return forces; }
        FloatType *getOldForces() { return oldForces; }
    };
}