#pragma once
#include <cstdint>

namespace ppb {

    struct PushHist {
        uint32_t numParticles;
        int32_t cCx;
        int32_t cCy;
        int32_t cCz;
        float bMinx;
        float bMiny;
        float bMinz;
        float bSizex;
        float bSizey;
        float bSizez;
    };

    struct PushBlelloch {
        uint32_t total_size;
        uint32_t block_size;
    };

    struct PushBlock {
        uint32_t total_size;
    };

    struct PushId {
        uint32_t numParticles;
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
        int32_t cCx;
        int32_t cCy;
        int32_t cCz;
    };

}
