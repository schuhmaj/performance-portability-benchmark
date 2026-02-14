#include "cuda.h"

// ============================================================================================
// This macro evalutes the return of a Cuda Driver API call and reports any returned
// error instead of failing silently. 
//
// Note:
//  - Some Cuda errors are reported asynchronously and may only surface on a later 
//    API call rather than the call where it originated.
//  - This macro reports errors but does not abort.
//
// This macro can be used with Cuda Driver API calls that return CUresult. 
// ============================================================================================

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
    // ============================================================================================
    // RAII structs to allocate- and free allocated Device memory / modules 
    // ============================================================================================

    struct DeviceMemory {
        CUdeviceptr ptr = 0;

        DeviceMemory() = default;

        DeviceMemory(size_t bytes) {
            CHECK(cuMemAlloc(&ptr, bytes));
        }

        void alloc(size_t bytes) {
            if (ptr) {
                CHECK(cuMemFree(ptr));
            }
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

    // ============================================================================================
    // Slang -> PTX memory layout matching
    //
    // When Slang compiles to PTX, each buffer pointer (CUdeviceptr) in the parameter block
    // is aligned to 16 bytes. This means that every 8 byte CUdeviceptr (pointing to a buffer)
    // is followed by 8 bytes of padding. 
    //
    // Note:
    //  - When compiling from Slang -> PTX, the output .ptx file will include a line similar to
    //      .const .align 8 .b8 SLANG_globalParams[40];
    //    When creating a struct to match the memory layout on the host side, make sure that
    //    the struct size matches the allocated memory in the .ptx file. In this case 40 bytes.
    //
    // Example (40 bytes):
    // Slang (KernelBlellochScan.slang):
    //  RWStructuredBuffer<uint> data;
    //  RWStructuredBuffer<uint> blockSums; 
    //  [push_constant] GlobalParams pc;
    // PTX:
    //  CUdeviceptr data; // 8 bytes
    //  8 bytes padding
    //  CUdeviceptr blockSums; // 8 bytes
    //  8 bytes padding
    //  CUdeviceptr pc;
    // Host:
    //  ResourceSlot data;
    //  ResrouceSlot blockSums;
    //  CUdeviceptr pc;
    //
    // ============================================================================================

    struct ResourceSlot {
        CUdeviceptr buffer_pointer;
        uint64_t padding; // required for 16 byte alignment
    };
    static_assert(sizeof(ResourceSlot) == 16, "ResourceSlot size mismatch!");

    // ============================================================================================
    // Recursive cache to store memory / modules for BlellochScan recursion.
    //
    // Note:
    //  - In the examples Cell Lists and Verlet Lists in nBodySimulation where this is used,
    //    the length of the exclusive scan is constant for a given invocation, meaning that 
    //    recursion depth and size of allocated memory remains static for each iteration.
    // ============================================================================================

    struct ExclusiveScanCache {
        DeviceMemory* pc;
        DeviceMemory* blockSum;

        DeviceModule* module_blellochScan;
        DeviceModule* module_blockSum;

        ExclusiveScanCache* child;
    };

    // ============================================================================================
    // Structs used for matching memory layout between host and device for specific shaders.
    // ============================================================================================

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
}
