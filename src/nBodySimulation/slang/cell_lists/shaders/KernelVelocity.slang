// KernelVelocity.slang
RWStructuredBuffer<float4> velocities;
StructuredBuffer<float4> forces;
StructuredBuffer<float4> oldForces;

struct GlobalParams
{
    float dt;
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

    float m = 1.0;

    float3 force = forces[gid].xyz + oldForces[gid].xyz;
    float t2m = pc.dt / (2.0 * m);
    float3 velChange = force * t2m;

    velocities[gid].xyz += velChange;
}
