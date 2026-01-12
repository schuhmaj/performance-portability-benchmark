// KernelPosition.slang
RWStructuredBuffer<float4> positions;
StructuredBuffer<float4> velocities;
RWStructuredBuffer<float4> forces;
RWStructuredBuffer<float4> oldForces;

uniform float4 globalForce;
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
    vector<float, 3> force = forces[i].xyz;
    oldForces[i].xyz = force;
    forces[i] = globalForce;

    float3 velocityPart = velocities[i].xyz * dt;
    float tt2m = (dt * dt) / (2.0 * m);
    float3 forcePart = force * tt2m;
    float3 displacement = velocityPart + forcePart;
    
    positions[i].xyz += displacement;
}
