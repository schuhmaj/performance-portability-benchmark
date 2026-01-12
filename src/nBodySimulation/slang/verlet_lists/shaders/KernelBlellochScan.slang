// KernelBlellochScan.slang
RWStructuredBuffer<uint> data;
RWStructuredBuffer<uint> blockSums;

uniform uint total_size;
uniform uint block_size;

groupshared uint tile[256];

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID, uint3 threadIdInGroup : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    uint gid = threadId.x;
    uint lid = threadIdInGroup.x;
    uint block = groupId.x;

    if (gid >= total_size) {
        return;
    }

    // -----------------------------------------------------------------------------------
    // load batch into shared memory
    // -----------------------------------------------------------------------------------
    tile[lid] = data[gid];
    GroupMemoryBarrierWithGroupSync();

    // -----------------------------------------------------------------------------------
    // upsweep
    // -----------------------------------------------------------------------------------
    for (uint off = 1; off < block_size; off <<= 1) {
        uint i = (lid + 1) * off * 2 - 1;
        if (i < block_size) {
            tile[i] += tile[i - off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == block_size - 1 && gid < total_size) {
        blockSums[block] = tile[lid];
    }

    // -----------------------------------------------------------------------------------
    // downsweep
    // -----------------------------------------------------------------------------------
    if (lid == block_size - 1) {
        tile[block_size - 1] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    for (uint off = block_size >> 1; off > 0; off >>= 1) {
        uint i = (lid + 1) * off * 2 - 1;
        if (i < block_size) {
            uint t = tile[i - off];
            tile[i - off] = tile[i];
            tile[i] += t;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // -----------------------------------------------------------------------------------
    // write back
    // -----------------------------------------------------------------------------------
    data[gid] = tile[lid];
}
