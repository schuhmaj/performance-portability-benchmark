#pragma once
#include <iostream>
#include <vector>
#include <utility>
#include <cublas_v2.h>

namespace ppb {
    template <typename FloatType>
    struct ImplCuda {
        using float_type = FloatType;
        cudaStream_t stream;
        ImplCuda();
        ~ImplCuda();
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a,
                                                             const std::vector<FloatType> &b);
    };

    template <typename FloatType>
    struct ImplChunkedCuda {
        using float_type = FloatType;
        static constexpr size_t NUM_STREAMS = 4;
        size_t chunkSize;
        cudaStream_t stream[NUM_STREAMS];
        ImplChunkedCuda();
        ~ImplChunkedCuda();
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a,
                                                     const std::vector<FloatType> &b);
    };

    template <typename FloatType>
    struct ImplCublas {
        using float_type = FloatType;
        cublasHandle_t handle;
        ImplCublas();
        ~ImplCublas();
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a,
                                                             const std::vector<FloatType> &b);
    };

    template <typename FloatType>
    struct ImplThrust {
        using float_type = FloatType;
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a,
                                                             const std::vector<FloatType> &b);
    };

    template <typename FloatType>
    struct ImplNvhpc {};

}
