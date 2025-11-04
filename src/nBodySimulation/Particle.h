#pragma once

#include <array>
#include <cmath>
#include <map>
#include <random>
#include <vector>
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

        Particle() = default;

        explicit Particle(std::array<FloatType, 3> position, const int type = 0)
            : _position{position}, _type{type} {}

        Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, const int type = 0)
            : _position{position}, _velocity{velocity}, _type{type} {}

        Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, std::array<FloatType, 3> force, const int type = 0)
            : _position{position}, _velocity{velocity}, _force{force}, _type{type} {}


        FloatType getMass() const {
            return 1.0;
        }
        FloatType getSigma() const {
            return 1.0;
        }
        FloatType getEpsilon() const {
            return 1.0;
        }

        int getType() const {
            return _type;
        }

        std::array<FloatType, 3> getPosition() const {
            return _position;
        }

        void setPosition(const std::array<FloatType, 3> &position) {
            _position = position;
        }

        void addPosition(const std::array<FloatType, 3> &position) {
            for (int i = 0; i < 3; ++i) {
                _position[i] += position[i];
            }
        }

        std::array<FloatType, 3> getVelocity() const {
            return _velocity;
        }

        void setVelocity(const std::array<FloatType, 3> &velocity) {
            _velocity = velocity;
        }

        void addVelocity(const std::array<FloatType, 3> &velocity) {
            for (int i = 0; i < 3; ++i) {
                _velocity[i] += velocity[i];
            }
        }

        std::array<FloatType, 3> getForce() const {
            return _force;
        }

        void setForce(const std::array<FloatType, 3> &force) {
            _force = force;
        }

        void addForce(const std::array<FloatType, 3> &force) {
            for (int i = 0; i < 3; ++i) {
                _force[i] += force[i];
            }
        }

        void subtractForce(const std::array<FloatType, 3> &force) {
            for (int i = 0; i < 3; ++i) {
                _force[i] -= force[i];
            }
        }

        std::array<FloatType, 3> getOldForce() const {
            return _oldForce;
        }

        void setOldForce(const std::array<FloatType, 3> &oldForce) {
            _oldForce = oldForce;
        }

        bool operator==(const Particle &other) const {
            return _type == other._type &&
                util::almostEqualRelative(_position[0], other._position[0]) &&
                util::almostEqualRelative(_position[1], other._position[1]) &&
                util::almostEqualRelative(_position[2], other._position[2]);
        }

        bool operator!=(const Particle &other) const {
            return !this->operator==(other);
        }

        static std::vector<Particle<FloatType>> generateUniform(const std::array<FloatType, 3> &boxMin, const std::array<FloatType, 3> &boxMax, size_t numParticles, const unsigned int seed = 42u) {
            std::vector<Particle<FloatType>> particles;
            particles.reserve(numParticles);
            std::mt19937 generator(seed);
            std::generate_n(std::back_inserter(particles), numParticles, [&]() -> Particle<FloatType> {
                return Particle<FloatType> {util::generatePosition<FloatType>(generator, boxMin, boxMax)};
            });
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
