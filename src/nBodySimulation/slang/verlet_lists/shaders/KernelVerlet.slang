// KernelVerlet.slang
StructuredBuffer<float4> positions;
RWStructuredBuffer<uint> verletLists;
StructuredBuffer<uint> starts;

struct GlobalParams
{
    uint numParticles;
    float radius;
};
[push_constant] GlobalParams pc;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint gid = threadId.x;
    if (gid >= pc.numParticles) { return; }

    uint index = starts[gid];
    uint end = starts[gid + 1];

    float3 pi = positions[gid].xyz;
    
    for (uint i = 0; i < pc.numParticles; i++) {
        if (i == gid) { continue; }
        if (index == end) { break; }
        // calculate distance to see if this particle is a neighbor
        float3 dr = pi - positions[i].xyz;
        float dr2 = dot(dr, dr);
        if (dr2 < pc.radius * pc.radius) {
            // if the particle is close enough, add it
            verletLists[index] = i;
            index += 1;
        }
    }
}
