// KernelPosition.slang
RWStructuredBuffer<float4> positions;
StructuredBuffer<float4> velocities;
RWStructuredBuffer<float4> forces;
RWStructuredBuffer<float4> oldForces;

uniform float globalForce_x;
uniform float globalForce_y;
uniform float globalForce_z;
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

    float3 force = forces[i].xyz;

    oldForces[i].xyz = force;

    forces[i].xyz = float3(globalForce_x, globalForce_y, globalForce_z);

    float3 velocityPart = velocities[i].xyz * dt;
    float tt2m = (dt * dt) / (2.0 * m);
    float3 forcePart = force * tt2m;
    float3 displacement = velocityPart + forcePart;

    positions[i].xyz += displacement.xyz;
}
