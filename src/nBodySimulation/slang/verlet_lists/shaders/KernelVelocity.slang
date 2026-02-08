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
[numthreads(1024,1,1)]
void computeVelocity(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= pc.numParticles) {
        return;
    }

    float m = 1.0;

    float3 force = forces[i].xyz + oldForces[i].xyz;
    float t2m = pc.dt / (2.0 * m);
    float3 velChange = force * t2m;

    velocities[i].xyz += velChange.xyz;
}
