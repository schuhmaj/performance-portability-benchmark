// FloatType / FloatType4 are injected via the build args (OPENCL_COMPILE_ARGS)
// according to the precision selected with PPB_FloatType.
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

__kernel void update_positions_reset_forces(
    __global FloatType4* positions,
    __global const FloatType4* velocities,
    __global FloatType4* forces,
    __global FloatType4* oldForces,
    const FloatType4 globalForce,
    const FloatType dt,
    const unsigned int numParticles
) {
    const int i = get_global_id(0);
    if (i >= numParticles) {
        return;
    }

    const FloatType m = 1.0;

    const FloatType4 force = forces[i];
    oldForces[i] = force;
    forces[i] = globalForce;

    const FloatType4 velocityPart = velocities[i] * dt;
    const FloatType tt2m = (dt * dt / (2.0 * m));
    const FloatType4 forcePart = force * tt2m;
    const FloatType4 displacement = velocityPart + forcePart;
    positions[i] += displacement;
}

__kernel void update_velocities(
    __global FloatType4* velocities,
    __global const FloatType4* forces,
    __global const FloatType4* oldForces,
    const FloatType dt,
    const unsigned int numParticles
) {
    const int i = get_global_id(0);
    if (i >= numParticles) {
        return;
    }

    const FloatType m = 1.0;

    const FloatType4 force = forces[i] + oldForces[i];
    const FloatType t2m = (dt / (2.0 * m));
    const FloatType4 velChange = force * t2m;
    velocities[i] += velChange;
}

__kernel void compute_forces(
    __global const FloatType4* positions,
    __global FloatType4* forces,
    const unsigned int numParticles
) {
    int i = get_global_id(0);
    if (i >= numParticles) {
        return;
    }

    const FloatType sigma = 1.0;
    const FloatType sigmaSquared = sigma * sigma;
    const FloatType epsilon24 = 1.0 * 24.0;

    const FloatType4 pi = positions[i];
    FloatType4 acc = (FloatType4)(0.0);

    for (int j = 0; j < numParticles; j++) {
        if (i == j) {
            continue;
        }

        const FloatType4 dr = pi - positions[j];
        const FloatType dr2 = dot(dr, dr);

        const FloatType invdr2 = 1.0 / dr2;
        FloatType lj6 = sigmaSquared * invdr2;
        lj6 = lj6 * lj6 * lj6;
        const FloatType lj12 = lj6 * lj6;
        const FloatType lj12m6 = lj12 - lj6;
        const FloatType fac = epsilon24 * (lj12 + lj12m6) * invdr2;
        acc += dr * fac;
    }
    forces[i] += acc;
}
