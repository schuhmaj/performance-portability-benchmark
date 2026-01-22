#pragma once
#include <vector>
#include <array>
#include <utility>
#include <chrono>
#include "omp.h"
#include "matrixMultiplication/MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    class ImplOpenMP {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,  const MatrixMultiplicationConfig &config);

    };

    };
