// KernelResetCells.slang
RWStructuredBuffer<uint> cells;

uniform uint total_size;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint gid = threadId.x;
    if (gid >= total_size) {
        return;
    }

    cells[gid] = 0;
}
