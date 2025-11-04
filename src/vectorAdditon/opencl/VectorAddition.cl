__kernel void add_vector_float(__global const float* a, __global const float* b, __global float* c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}

__kernel void add_vector_double(__global const double* a, __global const double* b, __global double* c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
