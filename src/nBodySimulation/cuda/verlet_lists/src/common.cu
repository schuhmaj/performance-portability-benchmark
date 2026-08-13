#include "common.cuh"
#include <iostream>

namespace ppb::cuda::nbody {
    //taken from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/ (last accessed 13.6.26, 19:44)
    void check(cudaError_t err, char const* func, char const* file, int line)
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                    << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        }
    }
} // namespace ppb::cuda::nbody