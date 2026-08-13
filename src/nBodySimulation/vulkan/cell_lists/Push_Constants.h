#pragma once
#include <cstdint>

// ============================================================================================
// Push Constant layouts used to sync between host and compute shaders
// 
// Each Struct represents a memory layout expected by specific shaders as Push Constants.
// The order and types of these constants needs to match the declaration of Push Constants
// in the shader code. These push data allocations are used in the backend for the vulkan 
// Cell Lists implementation.
// ============================================================================================

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

    struct PushId {
        uint32_t numParticles;
    };

    struct PushFor {
        uint32_t numParticles;
        int32_t cCx;
        int32_t cCy;
        int32_t cCz;
    };

}
