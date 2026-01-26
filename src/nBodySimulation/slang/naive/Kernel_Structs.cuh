#pragma once

namespace ppb {

    struct PushPos {
        // buffers
        float4* positions;
        float4* velocities;
        float4* forces;
        float4* oldForces;

        // push constants
        float globalForce_x;
        float globalForce_y;
        float globalForce_z;
        float dt;
        uint32_t numParticles;

        // padding
        uint32_t pad[5];
    };
    static_assert(sizeof(PushPos) == 72, "PushPos size mismatch!");

    struct PushVel {
        // buffers
        float4* velocities;
        float4* forces;
        float4* oldForces;

        // push constants
        float dt;
        uint32_t numParticles;

        // padding
        uint32_t pad[5];
    };
    static_assert(sizeof(PushVel) == 56, "PushPos size mismatch!");

    struct PushFor {
        // buffers
        float4* positions;
        float4* forces;

        // push constants
        uint32_t numParticles;

        // padding
        uint32_t pad[4];
    };
    static_assert(sizeof(PushFor) == 40, "PushPos size mismatch!");

}