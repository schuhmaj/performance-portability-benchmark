#pragma once
#include <cuda_runtime.h>
#include <iostream>

namespace ppb::cuda::nbody {
    //taken from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/ (last accessed 13.6.26, 19:44)
    void inline check(cudaError_t err, char const* func, char const* file, int line)
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                    << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        }
    }
    
    //taken from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/ (last accessed 13.6.26, 19:44)
    void inline checkLast(char const* file, int line) {
        cudaError_t const err{cudaGetLastError()};
        if (err != cudaSuccess)
        {
            std::cerr << "LAST CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
            std::cerr << cudaGetErrorString(err) << std::endl;
            // We don't exit when we encounter CUDA errors in this example.
            // std::exit(EXIT_FAILURE);
        }
    }
} // namespace ppb::cuda::nbody
