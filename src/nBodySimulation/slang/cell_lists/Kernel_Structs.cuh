#pragma once
#include "common/cuda/Common_Structs.cuh"

namespace ppb {

    // ============================================================================================
    // Structs used for matching memory layout between host and device.
    // ============================================================================================

    struct PushHist {
        // buffers
        ResourceSlot positions;
        ResourceSlot histogram;
        ResourceSlot particleIdx;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushHist) == 56, "PushHist size mismatch!");

    struct HistPushConstants {
        uint32_t numParticles;
        // cell counts
        int32_t cCount_x;
        int32_t cCount_y;
        int32_t cCount_z;
        // boxMin
        float bMin_x;
        float bMin_y;
        float bMin_z;
        // boxSize
        float bSize_x;
        float bSize_y;
        float bSize_z;
    };
    static_assert(sizeof(HistPushConstants) == 40, "HistPushConstants size mismatch!");


    struct PushId {
        // buffers
        ResourceSlot particleIdx;
        ResourceSlot idCells;
        ResourceSlot starts;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushId) == 56, "PushId size mismatch!");

    struct IdPushConstants {
        uint32_t numParticles;
    };
    static_assert(sizeof(IdPushConstants) == 4, "IdPushConstants size mismatch!");


    struct PushReset {
        // buffers
        ResourceSlot cells;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushReset) == 24, "PushReset size mismatch!");

    struct ResetPushConstants {
        uint32_t total_size;
    };
    static_assert(sizeof(ResetPushConstants) == 4, "ResetPushConstants size mismatch!");

    struct PushFor {
        // buffers
        ResourceSlot positions;
        ResourceSlot forces;
        ResourceSlot particleIdx;
        ResourceSlot starts;
        ResourceSlot idCells;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushFor) == 88, "PushPos size mismatch!");

    struct ForPushConstants {
        uint32_t numParticles;
        int32_t cCount_x;
        int32_t cCount_y;
        int32_t cCount_z;
    };
    static_assert(sizeof(ForPushConstants) == 16, "ForPushConstants size mismatch!");

};