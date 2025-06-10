#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <map>
#include "UtilityContainer.h"

#ifndef FUNCTION_PREFIX
#define FUNCTION_PREFIX
#endif

namespace ppb {

    template <typename FloatType>
    struct ParticleProperties {
        FloatType mass;
        FloatType sigma;
        FloatType epsilon;
    };

    template <typename FloatType>
    class Particle {

    public:
        std::array<FloatType, 3> _position{};
        std::array<FloatType, 3> _velocity{};
        std::array<FloatType, 3> _force{};
        std::array<FloatType, 3> _oldForce{};
        int _id{};

        static std::map<int, ParticleProperties<FloatType>> PARTICLE_PROPERTIES;

        FUNCTION_PREFIX Particle() {}

        FUNCTION_PREFIX Particle(std::array<FloatType, 3> position, int id = 0)
            : _position{position}, _id{id} {}

        FUNCTION_PREFIX Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, int id = 0)
            : _position{position}, _velocity{velocity}, _id{id} {}

        FUNCTION_PREFIX Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, std::array<FloatType, 3> force, int id = 0)
            : _position{position}, _velocity{velocity}, _force{force}, _id{id} {}


        FUNCTION_PREFIX FloatType getMass() const {
            return 1.0;
        }
        FUNCTION_PREFIX FloatType getSigma() const {
            return 1.0;
        }
        FUNCTION_PREFIX FloatType getEpsilon() const {
            return 5.0;
        }

        FUNCTION_PREFIX std::array<FloatType, 3> getPosition() const {
            return _position;
        }

        FUNCTION_PREFIX void setPosition(const std::array<FloatType, 3> &position) {
            _position = position;
        }

        FUNCTION_PREFIX void addPosition(const std::array<FloatType, 3> &position) {
            for (int i = 0; i < 3; ++i) {
                _position[i] += position[i];
            }
        }

        FUNCTION_PREFIX std::array<FloatType, 3> getVelocity() const {
            return _velocity;
        }

        FUNCTION_PREFIX void setVelocity(const std::array<FloatType, 3> &velocity) {
            _velocity = velocity;
        }

        FUNCTION_PREFIX void addVelocity(const std::array<FloatType, 3> &velocity) {
            for (int i = 0; i < 3; ++i) {
                _velocity[i] += velocity[i];
            }
        }

        FUNCTION_PREFIX std::array<FloatType, 3> getForce() const {
            return _force;
        }

        FUNCTION_PREFIX void setForce(const std::array<FloatType, 3> &force) {
            _force = force;
        }

        FUNCTION_PREFIX void addForce(const std::array<FloatType, 3> &force) {
            for (int i = 0; i < 3; ++i) {
                _force[i] += force[i];
            }
        }

        FUNCTION_PREFIX void subtractForce(const std::array<FloatType, 3> &force) {
            for (int i = 0; i < 3; ++i) {
                _force[i] -= force[i];
            }
        }

        FUNCTION_PREFIX std::array<FloatType, 3> getOldForce() const {
            return _oldForce;
        }

        FUNCTION_PREFIX void setOldForce(const std::array<FloatType, 3> &oldForce) {
            _oldForce = oldForce;
        }


        static std::vector<Particle<FloatType>> generateCuboid(const std::array<double, 3> &boxMin, const std::array<double, 3> &boxMax, size_t numParticles) {
            std::vector<Particle<FloatType>> particles;
            if (numParticles == 0) return particles;

            // Calculate lengths in each dimension
            double xLen = boxMax[0] - boxMin[0];
            double yLen = boxMax[1] - boxMin[1];
            double zLen = boxMax[2] - boxMin[2];

            // Find number of particles in x, y, z directions to fill the box as uniformly as possible
            size_t nX = std::max<size_t>(1, std::round(std::pow(numParticles * xLen * xLen / (yLen * zLen), 1.0/3)));
            size_t nY = std::max<size_t>(1, std::round(std::pow(numParticles * yLen * yLen / (xLen * zLen), 1.0/3)));
            size_t nZ = std::max<size_t>(1, numParticles / (nX * nY));
            if (nX * nY * nZ < numParticles) ++nZ;

            // Adjust counts to ensure we have at least numParticles, and min dimensions > 0
            while (nX * nY * nZ < numParticles) {
                if (nX <= nY && nX <= nZ) ++nX;
                else if (nY <= nZ) ++nY;
                else ++nZ;
            }

            // Compute grid spacings
            double dx = (nX > 1) ? xLen / (nX - 1) : 0.0;
            double dy = (nY > 1) ? yLen / (nY - 1) : 0.0;
            double dz = (nZ > 1) ? zLen / (nZ - 1) : 0.0;

            size_t count = 0;
            for (size_t ix = 0; ix < nX; ++ix) {
                for (size_t iy = 0; iy < nY; ++iy) {
                    for (size_t iz = 0; iz < nZ; ++iz) {
                        if (count >= numParticles) break;

                        std::array<FloatType, 3> pos = {
                            static_cast<FloatType>(boxMin[0] + ix * dx),
                            static_cast<FloatType>(boxMin[1] + iy * dy),
                            static_cast<FloatType>(boxMin[2] + iz * dz)
                        };
                        particles.emplace_back(pos, static_cast<FloatType>(1.0));
                        ++count;
                    }
                    if (count >= numParticles) break;
                }
                if (count >= numParticles) break;
            }
            return particles;
        }
    };
} // namespace ppb
