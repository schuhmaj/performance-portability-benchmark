#pragma once
#include <cstdint>

// ============================================================================================
// Push Constant layouts used to sync between host and compute shaders
// 
// Each Struct represents a memory layout expected by specific shaders as Push Constants.
// The order and types of these constants needs to match the declaration of Push Constants
// in the shader code. These push data allocations are used in the backend for the vulkan 
// Verlet Lists implementation.
// ============================================================================================

namespace ppb {

    struct PushCount {
        uint32_t numParticles;
        float radius;
    };

    struct PushVerlet {
        uint32_t numParticles;
        float radius;
    };

    struct PushFor {
        uint32_t numParticles;
    };

}
