// KernelVelocity.slang
RWStructuredBuffer<float4> velocities;
StructuredBuffer<float4> forces;
StructuredBuffer<float4> oldForces;

uniform float dt;
uniform uint numParticles;

[shader("compute")]
[numthreads(256,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= numParticles) {
        return;
    }

    float m = 1.0;

    float3 force = forces[i].xyz + oldForces[i].xyz;
    float t2m = dt / (2.0 * m);
    float3 velChange = force * t2m;

    velocities[i].xyz += velChange.xyz;
}
