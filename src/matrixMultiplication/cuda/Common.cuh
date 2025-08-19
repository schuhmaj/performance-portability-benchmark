
#pragma once

namespace ppb {

    template<typename T>
    __host__ __device__ inline T ceilDiv(const T &a, const T &b) {
        return (a + b - 1) / b;
    }

}