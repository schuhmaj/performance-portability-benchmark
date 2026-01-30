#pragma once

namespace ppb {

    struct ResourceSlot {
        CUdeviceptr buffer_pointer;
        uint64_t padding;
    };
    static_assert(sizeof(ResourceSlot) == 16, "ResourceSlot size mismatch!");


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
    static_assert(sizeof(HistPushConstants) == 40, "HistPushConstant size mismatch!");


    struct PusExclusive {
        // buffers
        ResourceSlot data;
        ResourceSlot blockSums;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PusExclusive) == 40, "PusExclusive size mismatch!");

    struct ExclusivePushConstants {
        uint32_t total_size;
        uint32_t block_size;
    };
    static_assert(sizeof(ExclusivePushConstants) == 8, "ExclusivePushConstant size mismatch!");


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
    static_assert(sizeof(IdPushConstants) == 4, "IdPushConstant size mismatch!");


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
    static_assert(sizeof(ResetPushConstants) == 4, "ResetPushConstant size mismatch!");


    struct PushPos {
        // buffers
        ResourceSlot positions;
        ResourceSlot velocities;
        ResourceSlot forces;
        ResourceSlot oldForces;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushPos) == 72, "PushPos size mismatch!");

    struct PosPushConstants {
        float globalForce_x;
        float globalForce_y;
        float globalForce_z;
        float dt;
        uint32_t numParticles;
    };
    static_assert(sizeof(PosPushConstants) == 20, "posPushConstant size mismatch!");


    struct PushVel {
        // buffers
        ResourceSlot velocities;
        ResourceSlot forces;
        ResourceSlot oldForces;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushVel) == 56, "PushPos size mismatch!");

    struct VelPushConstants {
        float dt;
        uint32_t numParticles;
    };
    static_assert(sizeof(VelPushConstants) == 8, "velPushConstants size mismatch!");


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