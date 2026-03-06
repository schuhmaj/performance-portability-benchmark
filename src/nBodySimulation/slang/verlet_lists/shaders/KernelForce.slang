// KernelForce.slang
StructuredBuffer<float4> positions;
RWStructuredBuffer<float4> forces;
StructuredBuffer<uint> verletLists;
StructuredBuffer<uint> starts;

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
    if (gid >= pc.numParticles) { return; }

    uint start = starts[gid];
    uint end = starts[gid + 1];

    float3 acc = float3(0.0);

    const float sigma = 1.0;
    const float sigmaSquared = sigma * sigma;
    const float epsilon24 = 24.0; // 1.0 * 24.0

    float3 pi = positions[gid].xyz;

    for (uint i = start; i < end; ++i) {
        uint j = verletLists[i];
        float3 dr = pi - positions[j].xyz;
        float dr2 = dot(dr, dr);

        float invdr2 = 1.0 / dr2;
        float lj6 = sigmaSquared * invdr2;
        lj6 = lj6 * lj6 * lj6;
        float lj12 = lj6 * lj6;
        float lj12m6 = lj12 - lj6;
        float fac = epsilon24 * (lj12 + lj12m6) * invdr2;

        acc += dr * fac;
    }

    forces[gid].xyz += acc;
}
