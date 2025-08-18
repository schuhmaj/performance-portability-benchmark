#pragma once
#include <vector>
#include <array>
#include "MatrixMultiplication.h"

namespace ppb {

    template<typename T>
    __host__ __device__ inline T ceilDiv(const T &a, const T &b) {
        return (a + b - 1) / b;
    }

    template<typename FloatType>
    class ImplCuda {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

        static dim3 getIdealBlockSize(unsigned int m, unsigned int n);

        static dim3 getIdealGridSize(const dim3 &blockSize, int m, int n);
    };

    };
