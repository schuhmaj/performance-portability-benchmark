// KernelHistogram.slang
StructuredBuffer<float4> positions;
RWStructuredBuffer<uint> histogram;
RWStructuredBuffer<int2> particleIdx;

struct GlobalParams
{
    uint numParticles;
    // cell counts
    int cCount_x;
    int cCount_y;
    int cCount_z;
    // boxMin
    float bMin_x;
    float bMin_y;
    float bMin_z;
    // boxSize
    float bSize_x;
    float bSize_y;
    float bSize_z;
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

    float3 pi = positions[gid].xyz;

    int gridIdx_x = int(clamp(int((pi.x - pc.bMin_x) / pc.bSize_x * pc.cCount_x), 0, pc.cCount_x - 1));
    int gridIdx_y = int(clamp(int((pi.y - pc.bMin_y) / pc.bSize_y * pc.cCount_y), 0, pc.cCount_y - 1));
    int gridIdx_z = int(clamp(int((pi.z - pc.bMin_z) / pc.bSize_z * pc.cCount_z), 0, pc.cCount_z - 1));
    int c = gridIdx_x + gridIdx_y * pc.cCount_x + gridIdx_z * pc.cCount_x * pc.cCount_y;

    int slot;
    InterlockedAdd<uint>(histogram[c], 1, slot);

    particleIdx[gid] = int2(c, slot);
}
