__kernel void update_positions_reset_forces(
    __global float4* positions,
    __global const float4* velocities,
    __global float4* forces,
    __global float4* oldForces,
    const float4 globalForce,
    const float dt,
    const unsigned int numParticles
) {
    const int i = get_global_id(0);
    if (i >= numParticles) {
        return;
    }

    const float m = 1.0;

    const float4 force = forces[i];
    oldForces[i] = force;
    forces[i] = globalForce;

    const float4 velocityPart = velocities[i] * dt;
    const float tt2m = (dt * dt / (2.0 * m));
    const float4 forcePart = force * tt2m;
    const float4 displacement = velocityPart + forcePart;
    positions[i] += displacement;
}

__kernel void update_velocities(
    __global float4* velocities,
    __global const float4* forces,
    __global const float4* oldForces,
    const float dt,
    const unsigned int numParticles
) {
    const int i = get_global_id(0);
    if (i >= numParticles) {
        return;
    }

    const float m = 1.0;

    const float4 force = forces[i] + oldForces[i];
    const float t2m = (dt / (2.0 * m));
    const float4 velChange = force * t2m;
    velocities[i] += velChange;
}

__kernel void compute_forces(
    __global const float4* positions,
    __global float4* forces,
    const unsigned int numParticles
) {
    int i = get_global_id(0);
    if (i >= numParticles) {
        return;
    }

    for (int j = 0; j < numParticles; j++) {
        if (i == j) {
            continue;
        }
        const float sigma = 1.0;
        const float sigmaSquared = sigma * sigma;
        const float epsilon24 = 1.0 * 24.0;

        const float4 dr = positions[i] - positions[j];
        const float dr2 = dot(dr, dr);

        const float invdr2 = 1.0 / dr2;
        float lj6 = sigmaSquared * invdr2;
        lj6 = lj6 * lj6 * lj6;
        const float lj12 = lj6 * lj6;
        const float lj12m6 = lj12 - lj6;
        const float fac = epsilon24 * (lj12 + lj12m6) * invdr2;
        const float4 force = dr * fac;
        forces[i] += force;
    }
}