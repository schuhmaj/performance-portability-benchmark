#pragma once
#include <vector>
#include <array>
#include "MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    class ImplCpp {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
