// KernelBlockSum.slang
RWStructuredBuffer<uint> data;
StructuredBuffer<uint> blockSums;

struct GlobalParams
{
    uint total_size;
};
[push_constant] GlobalParams pc;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    uint gid = threadId.x;
    uint block = groupId.x;

    if (gid >= pc.total_size) {
        return;
    }

    data[gid] += blockSums[block];
}
