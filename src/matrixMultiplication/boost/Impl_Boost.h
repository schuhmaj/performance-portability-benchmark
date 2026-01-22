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

        constexpr std::string_view kernel_name() const {
            if constexpr (std::is_same_v<FloatType, float>) {
                return "matrix_multiplication_float";
            } else if constexpr (std::is_same_v<FloatType, double>) {
                return "matrix_multiplication_double";
            }
            return "Unknown type";
        }

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
