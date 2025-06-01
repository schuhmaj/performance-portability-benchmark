#pragma once

#include <array>

namespace ppb {
    template <typename FloatType>
    class Particle {

    public:
        std::array<FloatType, 3> _position{};
        std::array<FloatType, 3> _velocity{};
        std::array<FloatType, 3> _force{};
        std::array<FloatType, 3> _oldForce{};
        FloatType _mass{};

        Particle(std::array<FloatType, 3> position, FloatType mass);

        Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, FloatType mass);

        Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, std::array<FloatType, 3> force,
                 FloatType mass);


    };
} // namespace ppb
