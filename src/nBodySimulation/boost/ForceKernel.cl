// Kernel 1: Update positions and reset forces
__kernel void update_positions_reset_forces(
    __global float4* positions,
    __global const float4* velocities,
    __global float4* forces,
    __global float4* oldForces,
    const float4 globalForce,
    const float dt,
    const int numParticles
) {
    int i = get_global_id(0);
    if (i >= numParticles) {
        return;
    }
    
    float m = 1.0f;
    float dt_half_m = dt * dt / (2.0f * m);
    
    float4 pos = positions[i];
    float4 vel = velocities[i];
    float4 force = forces[i];
    
    oldForces[i] = force;
    positions[i] = pos + vel * dt + force * dt_half_m;
    forces[i] = globalForce;
}
