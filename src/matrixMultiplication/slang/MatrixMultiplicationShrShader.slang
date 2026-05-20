// The TILE_SIZE matches the workgroup size (32x32)
#define TILE_SIZE 32

#ifndef FloatType
#define FloatType double
#define FloatType3 double3
#define FloatType4 double4
#define FloatTypeM dmat3
#endif

struct PushConsts {
    uint M;
    uint N;
    uint K;
};

// Use Vulkan attributes to map to push constants and bindings correctly
[[vk::push_constant]]
ConstantBuffer<PushConsts> pc;

[[vk::binding(0, 0)]] StructuredBuffer<FloatType> A;
[[vk::binding(1, 0)]] StructuredBuffer<FloatType> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<FloatType> C;

// Allocate shared memory for tiled matrix multiplication
groupshared FloatType tileA[TILE_SIZE][TILE_SIZE];
groupshared FloatType tileB[TILE_SIZE][TILE_SIZE];

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void matmul_kernel(
    uint3 global_id : SV_DispatchThreadID,
    uint3 local_id : SV_GroupThreadID
) {
    uint row = global_id.x;
    uint col = global_id.y;

    // Local indices within the 32x32 tile
    uint tx = local_id.x;
    uint ty = local_id.y;

    FloatType acc = (FloatType)0.0;

    // Calculate the number of tiles needed to cover the K dimension
    uint numTiles = (pc.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; ++t) {
        // 1. Collaboratively load a tile of matrix A into shared memory
        uint k_a = t * TILE_SIZE + ty;
        if (row < pc.M && k_a < pc.K) {
            tileA[tx][ty] = A[row + k_a * pc.M];
        } else {
            tileA[tx][ty] = (FloatType)0.0; // Pad out-of-bounds with zero
        }

        // 2. Collaboratively load a tile of matrix B into shared memory
        uint k_b = t * TILE_SIZE + tx;
        if (k_b < pc.K && col < pc.N) {
            tileB[tx][ty] = B[k_b + col * pc.K];
        } else {
            tileB[tx][ty] = (FloatType)0.0; // Pad out-of-bounds with zero
        }

        // Wait for all threads to finish loading their elements into shared memory
        GroupMemoryBarrierWithGroupSync();

        // 3. Compute the dot product for the current tile
        // Since we are iterating over the inner dimension, `k` sweeps across the tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            acc += tileA[tx][k] * tileB[k][ty];
        }

        // Wait for all threads to finish computing before loading the next tile
        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Write the final accumulated result to global memory
    if (row < pc.M && col < pc.N) {
        C[row + col * pc.M] = acc;
    }
}