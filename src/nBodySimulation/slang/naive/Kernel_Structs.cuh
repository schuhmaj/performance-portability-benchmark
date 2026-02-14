#pragma once
#include "common/cuda/Common_Structs.cuh"

namespace ppb {

    // ============================================================================================
    // Structs used for matching memory layout between host and device.
    // ============================================================================================

    struct PushFor {
        // buffers
        ResourceSlot positions;
        ResourceSlot forces;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushFor) == 40, "PushPos size mismatch!");

    struct ForPushConstants {
        uint32_t numParticles;
    };
    static_assert(sizeof(ForPushConstants) == 4, "forPushConstants size mismatch!");

}