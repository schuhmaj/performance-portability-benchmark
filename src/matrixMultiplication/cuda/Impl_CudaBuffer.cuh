#pragma once
#include <vector>
#include <array>
#include <utility>
#include "common/UtilityFloatArithmetic.h"
#include "matrixMultiplication/MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    class ImplCudaBuffer {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

        static dim3 getIdealBlockSize(unsigned int m, unsigned int n);

        static dim3 getIdealGridSize(const dim3 &blockSize, int m, int n);
    };

    };
