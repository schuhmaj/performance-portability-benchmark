#pragma once
#include <vector>
#include <array>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "MatrixMultiplication.h"

namespace ppb {

    template<typename FloatType>
    class ImplCublas {

        cublasHandle_t handle;

    public:

        using float_type = FloatType;
        using row_major = std::false_type;

        ImplCublas() {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("CUBLAS initialization failed");
            }
        }

        ~ImplCublas() {
            cublasDestroy(handle);
        }

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,  const MatrixMultiplicationConfig &config);

    };

    };
