#pragma once

namespace ppb {

    struct ResourceSlot {
        CUdeviceptr buffer_pointer;
        uint64_t padding;
    };
    static_assert(sizeof(ResourceSlot) == 16, "ResourceSlot size mismatch!");


    struct ExclusiveScanCache {
        CUdeviceptr pc;
        CUdeviceptr blockSum;
        CUmodule module_blellochScan;
        CUfunction kernel_blellochScan;
        CUmodule module_blockSum;
        CUfunction kernel_blockSum;

        ExclusiveScanCache* cache;
    };

    struct PushExclusive {
        // buffers
        ResourceSlot data;
        ResourceSlot blockSums;

        // push constants
        CUdeviceptr pc;
    };
    static_assert(sizeof(PushExclusive) == 40, "PushExclusive size mismatch!");

    struct ExclusivePushConstants {
        uint32_t total_size;
        uint32_t block_size;
    };
    static_assert(sizeof(ExclusivePushConstants) == 8, "ExclusivePushConstants size mismatch!");


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
    static_assert(sizeof(PosPushConstants) == 20, "PosPushConstants size mismatch!");


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
    static_assert(sizeof(VelPushConstants) == 8, "VelPushConstants size mismatch!");


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