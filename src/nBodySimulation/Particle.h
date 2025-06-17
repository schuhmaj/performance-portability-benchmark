#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <map>
#include "UtilityContainer.h"
#include "UtilityFloatArithmetic.h"

namespace ppb {

    template <typename FloatType>
    struct ParticleProperties {
        FloatType mass;
        FloatType sigma;
        FloatType epsilon;
    };

    template <typename FloatType>
    class Particle {

        std::array<FloatType, 3> _position{};
        std::array<FloatType, 3> _velocity{};
        std::array<FloatType, 3> _force{};
        std::array<FloatType, 3> _oldForce{};
        int _type{};

    public:


        static std::map<int, ParticleProperties<FloatType>> PARTICLE_PROPERTIES;

        __attribute__((device)) __attribute__((host)) Particle() = default;

        __attribute__((device)) __attribute__((host)) explicit Particle(std::array<FloatType, 3> position, const int type = 0)
            : _position{position}, _type{type} {}

        __attribute__((device)) __attribute__((host)) Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, const int type = 0)
            : _position{position}, _velocity{velocity}, _type{type} {}

        __attribute__((device)) __attribute__((host)) Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, std::array<FloatType, 3> force, const int type = 0)
            : _position{position}, _velocity{velocity}, _force{force}, _type{type} {}


        __attribute__((device)) __attribute__((host)) FloatType getMass() const {
            return 1.0;
        }
        __attribute__((device)) __attribute__((host)) FloatType getSigma() const {
            return 1.0;
        }
        __attribute__((device)) __attribute__((host)) FloatType getEpsilon() const {
            return 5.0;
        }

        __attribute__((device)) __attribute__((host)) int getType() const {
            return _type;
        }

        __attribute__((device)) __attribute__((host)) std::array<FloatType, 3> getPosition() const {
            return _position;
        }

        __attribute__((device)) __attribute__((host)) void setPosition(const std::array<FloatType, 3> &position) {
            _position = position;
        }

        __attribute__((device)) __attribute__((host)) void addPosition(const std::array<FloatType, 3> &position) {
            for (int i = 0; i < 3; ++i) {
                _position[i] += position[i];
            }
        }

        __attribute__((device)) __attribute__((host)) std::array<FloatType, 3> getVelocity() const {
            return _velocity;
        }

        __attribute__((device)) __attribute__((host)) void setVelocity(const std::array<FloatType, 3> &velocity) {
            _velocity = velocity;
        }

        __attribute__((device)) __attribute__((host)) void addVelocity(const std::array<FloatType, 3> &velocity) {
            for (int i = 0; i < 3; ++i) {
                _velocity[i] += velocity[i];
            }
        }

        __attribute__((device)) __attribute__((host)) std::array<FloatType, 3> getForce() const {
            return _force;
        }

        __attribute__((device)) __attribute__((host)) void setForce(const std::array<FloatType, 3> &force) {
            _force = force;
        }

        __attribute__((device)) __attribute__((host)) void addForce(const std::array<FloatType, 3> &force) {
            for (int i = 0; i < 3; ++i) {
                _force[i] += force[i];
            }
        }

        __attribute__((device)) __attribute__((host)) void subtractForce(const std::array<FloatType, 3> &force) {
            for (int i = 0; i < 3; ++i) {
                _force[i] -= force[i];
            }
        }

        __attribute__((device)) __attribute__((host)) std::array<FloatType, 3> getOldForce() const {
            return _oldForce;
        }

        __attribute__((device)) __attribute__((host)) void setOldForce(const std::array<FloatType, 3> &oldForce) {
            _oldForce = oldForce;
        }

        bool operator==(const Particle &other) const {
            return _type == other._type &&
                util::almostEqualRelative(_position[0], other._position[0]) &&
                util::almostEqualRelative(_position[1], other._position[1]) &&
                util::almostEqualRelative(_position[2], other._position[2]) &&
                util::almostEqualRelative(_velocity[0], other._velocity[0]) &&
                util::almostEqualRelative(_velocity[1], other._velocity[1]) &&
                util::almostEqualRelative(_velocity[2], other._velocity[2]) &&
                util::almostEqualRelative(_force[0], other._force[0]) &&
                util::almostEqualRelative(_force[1], other._force[1]) &&
                util::almostEqualRelative(_force[2], other._force[2]);
        }

        bool operator!=(const Particle &other) const {
            return !this->operator==(other);
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

    template<typename FloatType>
    std::ostream& operator<<(std::ostream& os, const Particle<FloatType>& particle) {
        const auto& pos = particle.getPosition();
        const auto& vel = particle.getVelocity();
        const auto& force = particle.getForce();
        const auto& oldForce = particle.getOldForce();
        os << "Particle(type=" << particle.getType()
           << ", position=[" << pos[0] << ", " << pos[1] << ", " << pos[2] << "]"
           << ", velocity=[" << vel[0] << ", " << vel[1] << ", " << vel[2] << "]"
           << ", force=[" << force[0] << ", " << force[1] << ", " << force[2] << "]"
           << ", oldForce=[" << oldForce[0] << ", " << oldForce[1] << ", " << oldForce[2] << "])";
        return os;
    }
} // namespace ppb
