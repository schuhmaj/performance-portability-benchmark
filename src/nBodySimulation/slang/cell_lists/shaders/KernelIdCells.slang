// KernelIdCells.slang
StructuredBuffer<int2> particleIdx;
RWStructuredBuffer<uint> idCells;
StructuredBuffer<uint> starts;


struct GlobalParams
{
    uint numParticles;
};
[push_constant] GlobalParams pc;

[shader("compute")]
[numthreads(1024,1,1)]
void computeIdCells(uint3 threadId : SV_DispatchThreadID)
{
    uint gid = threadId.x;
    if (gid >= pc.numParticles) {
        return;
    }

    int2 data = particleIdx[gid];
    int cell = data.x;
    int slot = data.y;

    uint idx = starts[cell] + slot;

    idCells[idx] = gid;
}
