#pragma once
#include <vector>
#include <array>
#include "MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    class ImplCublas {

    public:

        using float_type = FloatType;

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,  const MatrixMultiplicationConfig &config);

    };

    };
