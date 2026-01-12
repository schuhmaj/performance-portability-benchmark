// KernelCountNeighbors.slang
StructuredBuffer<float4> positions;
RWStructuredBuffer<uint> nNeighbors;

uniform uint n;
uniform float radius;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint gid = threadId.x;

    if (gid >= n) {
        return;
    }
    
    float3 pi = positions[gid].xyz;

    uint neighbors = 0;
    for (uint i = 0; i < n; i++) {
        if (i == gid) { continue; }
        // calculate distance to see if this particle is a neighbor
        float3 dr = pi - positions[i].xyz;
        float dr2 = dot(dr.xyz, dr.xyz);
        if (dr2 < radius * radius) {
            // if the particle is close enough, add it
            neighbors += 1;
        }
    }
    
    nNeighbors[gid] = neighbors;
}
