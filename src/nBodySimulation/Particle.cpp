
#include "Particle.h"

namespace ppb {

    template<typename FloatType>
    Particle<FloatType>::Particle(std::array<FloatType, 3> position, FloatType mass) : _position{position}, _mass{mass} {}

    template<typename FloatType>
    Particle<FloatType>::Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, FloatType mass) : _position{position}, _velocity{velocity}, _mass{mass} {}

    template<typename FloatType>
    Particle<FloatType>::Particle(std::array<FloatType, 3> position, std::array<FloatType, 3> velocity, std::array<FloatType, 3> force, FloatType mass) : _position{position}, _velocity{velocity}, _force{force}, _mass{mass} {}

    template class Particle<float>;
    template class Particle<double>;

} //nnamespace ppb