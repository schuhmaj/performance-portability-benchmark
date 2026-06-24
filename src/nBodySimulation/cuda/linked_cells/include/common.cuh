#pragma once

namespace ppb {
    extern void check(cudaError_t err, char const* func, char const* file, int line);
} // namespace ppb