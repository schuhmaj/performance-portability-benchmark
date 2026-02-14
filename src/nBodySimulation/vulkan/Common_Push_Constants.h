#pragma once
#include <cstdint>

// ============================================================================================
// Push Constant layouts used to sync between host and compute shaders
// 
// Each Struct represents a memory layout expected by specific shaders as Push Constants.
// The order and types of these constants needs to match the declaration of Push Constants
// in the shader code. These push data allocations are reused between different backends
// of nBodySimulation implementations.
// ============================================================================================

namespace ppb {

    struct PushBlelloch {
        uint32_t total_size;
        uint32_t block_size;
    };

    struct PushBlock {
        uint32_t total_size;
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

}
