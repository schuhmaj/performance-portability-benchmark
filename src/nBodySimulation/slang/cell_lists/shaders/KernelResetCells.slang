// KernelResetCells.slang
RWStructuredBuffer<uint> cells;

struct GlobalParams
{
    uint total_size;
};
[push_constant] GlobalParams pc;

[shader("compute")]
[numthreads(256,1,1)]
void computeResetCells(uint3 threadId : SV_DispatchThreadID)
{
    uint gid = threadId.x;
    if (gid >= pc.total_size) {
        return;
    }
    
    cells[gid] = 0;
}
