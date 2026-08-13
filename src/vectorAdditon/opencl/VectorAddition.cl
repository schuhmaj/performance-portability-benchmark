__kernel void add_vector(__global const FloatType* a, __global const FloatType* b, __global FloatType* c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
}
