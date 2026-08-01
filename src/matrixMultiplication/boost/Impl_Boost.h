#pragma once
#include <vector>
#include <array>
#include <utility>
#include "boost/compute.hpp"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "MatrixMultiplicationKernel.h"

namespace ppb {

    template<typename FloatType>
    class ImplBoost {

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        boost::compute::device gpu;
        boost::compute::context context;
        boost::compute::command_queue queue;
        boost::compute::program program;
        boost::compute::kernel kernel;

        ImplBoost();

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
