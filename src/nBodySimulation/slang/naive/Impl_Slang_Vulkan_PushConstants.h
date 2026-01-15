#pragma once
#include <cstdint>

namespace ppb {

    struct PushPos {
        float globalForce_x;
        float globalForce_y;
        float globalForce_z;
        float dt;
        uint32_t numParticles;
    };

    struct PushVel {
        float dt;
        uint32_t numParticles;
    };

    struct PushFor {
        uint32_t numParticles;
    };

}
