#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * Checks the return value of a HIP runtime API call and aborts with a
 * descriptive message if it did not return hipSuccess.
 *
 * Usage: CHECK_HIP(hipMalloc(&ptr, bytes));
 */
#define CHECK_HIP(call)                                                                       \
    do {                                                                                      \
        const hipError_t _err = (call);                                                       \
        if (_err != hipSuccess) {                                                             \
            std::fprintf(stderr, "HIP error %s:%d: '%s' returned %d (%s)\n",                  \
                         __FILE__, __LINE__, #call, static_cast<int>(_err),                   \
                         hipGetErrorString(_err));                                            \
            std::abort();                                                                     \
        }                                                                                     \
    } while (0)
