// KernelForce.slang
StructuredBuffer<float4> positions;
RWStructuredBuffer<float4> forces;
StructuredBuffer<int2> particleIdx;
StructuredBuffer<uint> starts;
StructuredBuffer<uint> idCells;

uniform uint numParticles;
uniform int cCount_x;
uniform int cCount_y;
uniform int cCount_z;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= numParticles) {
        return;
    }
    float3 acc = float3(0.0);

    const float sigma = 1.0;
    const float sigmaSquared = sigma * sigma;
    const float epsilon24 = 24.0; // 1.0 * 24.0

    float3 pi = positions[i].xyz;
    int ccxy = cCount_x * cCount_y;
    int ccxyz = ccxy * cCount_z;
    int c = particleIdx[i].x;
    int cz = c / ccxy;
    int rem = c % ccxy;
    int cy = rem / cCount_x;
    int cx = rem % cCount_x;

    for (int dx = -1; dx <= 1; dx++) {
        int nx = cx + dx;
        if (nx >= 0 && nx < cCount_x) {
            for (int dy = -1; dy <= 1; dy++) {
                int ny = cy + dy;
                if (ny >= 0 && ny < cCount_y) {
                    for (int dz = -1; dz <= 1; dz++) {
                        int nz = cz + dz;
                        if (nz >= 0 && nz < cCount_z) {
                            int cell = nx + ny * cCount_x + nz * ccxy;
                            for (uint u = starts[cell]; u < starts[cell + 1]; u++) {
                                uint j = idCells[u];
                                if (i == j) {
                                    continue;
                                }
                                float3 dr = pi - positions[j].xyz;
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
            }
        }
    }

    forces[i].xyz += acc.xyz;
}
