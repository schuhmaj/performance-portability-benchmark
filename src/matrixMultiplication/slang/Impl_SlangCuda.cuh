#pragma once
#include <vector>
#include <array>
#include <utility>
#include "common/cuda/Common_Structs.cuh"
#include "matrixMultiplication/MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    class ImplSlangCuda {

        CudaContext context;
        DeviceModule module;

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        ImplSlangCuda();

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config);

    };

    };
