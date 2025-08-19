
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include "Impl_Cublas.cuh"

namespace ppb {
    template <typename FloatType>
    std::vector<FloatType> ImplCublas<FloatType>::operator()(const std::vector<FloatType> &a,
                                                             const std::vector<FloatType> &b,  const MatrixMultiplicationConfig &config) {
        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("CUBLAS initialization failed");
        }

        // Allocate device memory
        FloatType *devA, *devB, *devC;
        const size_t sizeA = config.m * config.k * sizeof(FloatType);
        const size_t sizeB = config.k * config.n * sizeof(FloatType);
        const size_t sizeC = config.m * config.n * sizeof(FloatType);

        cudaMalloc(&devA, sizeA);
        cudaMalloc(&devB, sizeB);
        cudaMalloc(&devC, sizeC);

        cudaMemcpy(devA, a.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, b.data(), sizeB, cudaMemcpyHostToDevice);

        // Set up GEMM parameters
        const FloatType alpha = 1.0f;
        const FloatType beta = 0.0f;

        // Perform matrix multiplication: C = alpha * A * B + beta * C
        // Note: CUBLAS expects column-major matrices
        if constexpr (std::is_same_v<FloatType, float>) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       config.m, config.n, config.k,
                       &alpha,
                       devA, config.m,
                       devB, config.k,
                       &beta,
                       devC, config.m);
        } else if constexpr (std::is_same_v<FloatType, double>) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            config.m, config.n, config.k,
            &alpha,
            devA, config.m,
            devB, config.k,
            &beta,
            devC, config.m);
        } else {
            static_assert(std::is_same_v<FloatType, float> || std::is_same_v<FloatType, double>, "Unsupported type");
        }

        std::vector<FloatType> result(config.m * config.n);
        cudaMemcpy(result.data(), devC, sizeC, cudaMemcpyDeviceToHost);

        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);

        cublasDestroy(handle);
        return result;
    }

    // Explicit instantiation for float and double
    template class ImplCublas<float>;
    template class ImplCublas<double>;
}