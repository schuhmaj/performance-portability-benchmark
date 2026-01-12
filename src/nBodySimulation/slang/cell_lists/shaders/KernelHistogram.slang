// KernelHistogram.slang
StructuredBuffer<float4> positions;
RWStructuredBuffer<uint> histogram;
RWStructuredBuffer<int2> particleIdx;

uniform uint numParticles;
// cell counts
uniform int cCount_x;
uniform int cCount_y;
uniform int cCount_z;
// boxMin
uniform float bMin_x;
uniform float bMin_y;
uniform float bMin_z;
// boxSize
uniform float bSize_x;
uniform float bSize_y;
uniform float bSize_z;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= numParticles) {
        return;
    }

    float3 pi = positions[i].xyz;

    int gridIdx_x = int(clamp(int((pi.x - bMin_x) / bSize_x * cCount_x), 0, cCount_x - 1));
    int gridIdx_y = int(clamp(int((pi.y - bMin_y) / bSize_y * cCount_y), 0, cCount_y - 1));
    int gridIdx_z = int(clamp(int((pi.z - bMin_z) / bSize_z * cCount_z), 0, cCount_z - 1));
    int c = gridIdx_x + gridIdx_y * cCount_x + gridIdx_z * cCount_x * cCount_y;

    int slot;
    InterlockedAdd<uint>(histogram[c], 1, slot);
    particleIdx[i] = int2(c, slot);
}
