// KernelIdCells.slang
StructuredBuffer<int2> particleIdx;
RWStructuredBuffer<uint> idCells;
StructuredBuffer<uint> starts;

uniform uint numParticles;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= numParticles) {
        return;
    }

    int2 data = particleIdx[i];
    int cell = data.x;
    int slot = data.y;

    uint idx = starts[cell] + slot;

    idCells[idx] = i;
}
