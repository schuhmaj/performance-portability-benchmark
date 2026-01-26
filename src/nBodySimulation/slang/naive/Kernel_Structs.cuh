#pragma once

namespace ppb {

    struct alignas(8) PushPos {
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
        uint32_t pad[3];
    };

    struct alignas(8) PushVel {
        // buffers
        float4* velocities;
        float4* forces;
        float4* oldForces;

        // push constants
        float dt;
        uint32_t numParticles;

        // padding
        uint32_t pad[3];
    };

    struct alignas(8) PushFor {
        // buffers
        float4* positions;
        float4* forces;

        // push constants
        uint32_t numParticles;

        // padding
        uint32_t pad;
    };

}