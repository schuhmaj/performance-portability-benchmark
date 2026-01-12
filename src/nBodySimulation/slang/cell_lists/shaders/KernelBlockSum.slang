// KernelBlockSum.slang
RWStructuredBuffer<uint> data;
StructuredBuffer<uint> blockSums;

uniform uint total_size;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    uint gid = threadId.x;
    uint block = groupId.x;

    if (gid >= total_size) {
        return;
    }

    data[gid] += blockSums[block];
}
