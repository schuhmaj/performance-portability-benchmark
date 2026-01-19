#pragma once
#include <cstdint>

namespace ppb {

    struct PushCount {
        uint32_t numParticles;
        float radius;
    };

    struct PushBlelloch {
        uint32_t total_size;
        uint32_t block_size;
    };

    struct PushBlock {
        uint32_t total_size;
    };

    struct PushVerlet {
        uint32_t numParticles;
        float radius;
    };

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
