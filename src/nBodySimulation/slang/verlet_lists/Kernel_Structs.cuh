#pragma once
#include "common/cuda/Common_Structs.cuh"

namespace ppb {

    // ============================================================================================
    // Structs used for matching memory layout between host and device.
    // ============================================================================================

    struct PushCount {
        // buffers
        ResourceSlot positions;
        ResourceSlot nNeighbors;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushCount) == 40, "PushCount size mismatch!");

    struct CountPushConstants {
        uint32_t n;
        float radius;
    };
    static_assert(sizeof(CountPushConstants) == 8, "CountPushConstants size mismatch!");


    struct PushVerlet {
        // buffers
        ResourceSlot positions;
        ResourceSlot verletLists;
        ResourceSlot neighborsStarts;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushVerlet) == 56, "PushVerlet size mismatch!");

    struct VerletPushConstants {
        uint32_t total_size;
        float radius;
    };
    static_assert(sizeof(VerletPushConstants) == 8, "VerletPushConstants size mismatch!");


    struct PushFor {
        // buffers
        ResourceSlot positions;
        ResourceSlot forces;
        ResourceSlot verletLists;
        ResourceSlot neighborsStarts;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushFor) == 72, "PushPos size mismatch!");

    struct ForPushConstants {
        uint32_t numParticles;
    };
    static_assert(sizeof(ForPushConstants) == 4, "ForPushConstants size mismatch!");

}