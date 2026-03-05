// KernelForce.slang
StructuredBuffer<float4> positions;
RWStructuredBuffer<float4> forces;
StructuredBuffer<int2> particleIdx;
StructuredBuffer<uint> starts;
StructuredBuffer<uint> idCells;
StructuredBuffer<float4> particleCells;

struct GlobalParams
{
    uint numParticles;
    int cCount_x;
    int cCount_y;
    int cCount_z;
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
    float3 acc = float3(0.0);

    const float sigma = 1.0;
    const float sigmaSquared = sigma * sigma;
    const float epsilon24 = 24.0; // 1.0 * 24.0

    float3 pi = positions[gid].xyz;
    int ccxy = pc.cCount_x * pc.cCount_y;
    int ccxyz = ccxy * pc.cCount_z;
    int c = particleIdx[gid].x;
    int cz = c / ccxy;
    int rem = c % ccxy;
    int cy = rem / pc.cCount_x;
    int cx = rem % pc.cCount_x;

    for (int dz = -1; dz <= 1; dz++) {
        int nz = cz + dz;
        if (nz < 0 || nz >= pc.cCount_z) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cy + dy;
            if (ny < 0 || ny >= pc.cCount_y) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx;
                if (nx < 0 || nx >= pc.cCount_x) continue;
                int cell = nx + ny * pc.cCount_x + nz * ccxy;
                for (uint u = starts[cell]; u < starts[cell + 1]; u++) {
                    uint j = idCells[u];
                    if (gid == j) continue;
                    float3 dr = pi - particleCells[u].xyz;
                    float dr2 = dot(dr.xyz, dr.xyz);

                    float invdr2 = 1.0 / dr2;
                    float lj6 = sigmaSquared * invdr2;
                    lj6 = lj6 * lj6 * lj6;
                    float lj12 = lj6 * lj6;
                    float lj12m6 = lj12 - lj6;
                    float fac = epsilon24 * (lj12 + lj12m6) * invdr2;

                    acc += dr * fac;
                }
            }
        }
    }

    forces[gid].xyz += acc.xyz;
}
