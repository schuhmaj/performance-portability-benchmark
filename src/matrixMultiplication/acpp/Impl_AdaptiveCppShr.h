#pragma once
#include <vector>
#include <array>
#include <utility>
#include <sycl/sycl.hpp>
#include "matrixMultiplication/MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    class ImplAdaptiveCppShr {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;
        static constexpr size_t ALIGNMENT = 64;

        sycl::queue queue;

        ImplAdaptiveCppShr();

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
