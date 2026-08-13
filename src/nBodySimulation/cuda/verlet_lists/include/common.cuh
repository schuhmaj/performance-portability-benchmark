#pragma once

namespace ppb::cuda::nbody {
    extern void check(cudaError_t err, char const* func, char const* file, int line);
} // namespace ppb::cuda::nbody