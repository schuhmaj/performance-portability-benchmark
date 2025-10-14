#pragma once

#include "Particle.h"

namespace ppb {
    template <typename FloatType>
    class ParticleContainer {
        std::vector<FloatType> positions;
        std::vector<FloatType> velocities;
        std::vector<FloatType> forces;
        std::vector<FloatType> oldForces;

    public:
        ParticleContainer() = default;
    
        explicit ParticleContainer(const std::vector<Particle<FloatType>> &particles) {
            positions.resize(particles.size() * 3);
            velocities.resize(particles.size() * 3);
            forces.resize(particles.size() * 3);
            oldForces.resize(particles.size() * 3);
            for (size_t i = 0; i < particles.size(); ++i) {
                std::copy(particles[i].getPosition().begin(), particles[i].getPosition().end(), positions.begin() + 3 * i);
                std::copy(particles[i].getVelocity().begin(), particles[i].getVelocity().end(), velocities.begin() + 3 * i);
                std::copy(particles[i].getForce().begin(), particles[i].getForce().end(), forces.begin() + 3 * i);
                std::copy(particles[i].getOldForce().begin(), particles[i].getOldForce().end(), oldForces.begin() + 3 * i);
            }
        }

        void extractParticleData(std::vector<Particle<FloatType>> &particles) {
            for (size_t i = 0; i < particles.size(); ++i) {
                std::copy(positions.begin() + 3 * i, positions.begin() + 3 * i + 3, particles[i].getPosition().begin());
                std::copy(velocities.begin() + 3 * i, velocities.begin() + 3 * i + 3, particles[i].getVelocity().begin());
                std::copy(forces.begin() + 3 * i, forces.begin() + 3 * i + 3, particles[i].getForce().begin());
                std::copy(oldForces.begin() + 3 * i, oldForces.begin() + 3 * i + 3, particles[i].getOldForce().begin());
            }
        }

        std::vector<FloatType> *getPositions() { return &positions; }
        std::vector<FloatType> *getVelocities() { return &velocities; }
        std::vector<FloatType> *getForces() { return &forces; }
        std::vector<FloatType> *getOldForces() { return &oldForces; }
    };
}