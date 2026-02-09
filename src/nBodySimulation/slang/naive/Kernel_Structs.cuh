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