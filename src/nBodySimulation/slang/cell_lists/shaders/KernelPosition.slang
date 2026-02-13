// KernelPosition.slang
RWStructuredBuffer<float4> positions;
StructuredBuffer<float4> velocities;
RWStructuredBuffer<float4> forces;
RWStructuredBuffer<float4> oldForces;

struct GlobalParams
{
    float globalForce_x;
    float globalForce_y;
    float globalForce_z;
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

    float3 force = forces[gid].xyz;

    oldForces[gid].xyz = force;

    forces[gid].xyz = float3(pc.globalForce_x, pc.globalForce_y, pc.globalForce_z);

    float3 velocityPart = velocities[gid].xyz * pc.dt;
    float tt2m = (pc.dt * pc.dt) / (2.0 * m);
    float3 forcePart = force * tt2m;
    float3 displacement = velocityPart + forcePart;

    positions[gid].xyz += displacement.xyz;
}
