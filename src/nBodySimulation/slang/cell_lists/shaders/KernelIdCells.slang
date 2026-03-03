// KernelIdCells.slang
StructuredBuffer<float4> positions;
StructuredBuffer<int2> particleIdx;
StructuredBuffer<uint> starts;
RWStructuredBuffer<uint> idCells;
RWStructuredBuffer<float4> particleCells;

struct GlobalParams
{
    uint numParticles;
};
[push_constant] GlobalParams pc;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
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
    particleCells[idx].xyz = positions[gid].xyz;
}
