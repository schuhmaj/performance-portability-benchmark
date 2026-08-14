#pragma once

namespace ppb::cuda::nbody {
    __device__ inline float3 make_float3_add(const float3 a, const float3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ inline float3 make_float3_sub(const float3 a, const float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ inline float3 make_float3_scale(const float3 v, const float s) {
        return make_float3(v.x * s, v.y * s, v.z * s);
    }

    __device__ inline float dot3(const float3 a, const float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    } 
} // namespace ppb::cuda::nbody
