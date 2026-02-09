#pragma once

#define CHECK(X)                                                       \
    do {                                                               \
        CUresult err = (X);                                            \
        if (err != CUDA_SUCCESS) {                                     \
            const char* msg;                                           \
            cuGetErrorString(err, &msg);                               \
            fprintf(stderr,                                            \
                "CUDA Driver error at %s:%d (%s): %s\n",               \
                __FILE__, __LINE__, #X, msg);                          \
        }                                                              \
    } while (0)


namespace ppb {

    struct DeviceMemory {
        CUdeviceptr ptr = 0;

        DeviceMemory(size_t bytes) {
            CHECK(cuMemAlloc(&ptr, bytes));
        }
        ~DeviceMemory() {
            if (ptr) {
                CHECK(cuMemFree(ptr));
            }
        }
    };

    struct DeviceModule {
        CUmodule mod      = nullptr;
        CUfunction kernel = nullptr;

        ~DeviceModule() {
            if (mod) {
                CHECK(cuModuleUnload(mod));
            }
        }
    };

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
    static_assert(sizeof(HistPushConstants) == 40, "HistPushConstants size mismatch!");


    struct ExclusiveScanCache {
        DeviceMemory* pc;
        DeviceMemory* blockSum;

        DeviceModule* module_blellochScan;
        DeviceModule* module_blockSum;

        ExclusiveScanCache* child;
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