#include "vectorAdditon/cuda/Implementations.cuh"
#include "vectorAdditon/VectorAddition.h"
#include <benchmark/benchmark.h>


namespace ppb {
    template <typename FloatType>
    ImplCublas<FloatType>::ImplCublas()
    : handle {nullptr} {
        cublasCreate(&handle);
    }

    template <typename FloatType>
    ImplCublas<FloatType>::~ImplCublas() {
        cublasDestroy(handle);
    }

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplCublas<FloatType>::operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        int err = 0;
        const size_t size = a.size();
        FloatType* deviceA;
        FloatType* deviceC;

        cudaMalloc(&deviceA, size * sizeof(FloatType));
        cudaMalloc(&deviceC, size * sizeof(FloatType));
        cudaMemcpy(deviceA, a.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceC, b.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice);

        constexpr FloatType alpha = 1.0;
        if constexpr (std::is_same_v<FloatType, float>) {
            cudaEventRecord(start);
            err = cublasSaxpy(handle, size, &alpha, deviceA, 1, deviceC, 1);
            cudaEventRecord(stop);
        } else if constexpr (std::is_same_v<FloatType, double>) {
            cudaEventRecord(start);
            err = cublasDaxpy(handle, size, &alpha, deviceA, 1, deviceC, 1);
            cudaEventRecord(stop);
        }
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);

        std::vector<FloatType> result(size);
        cudaMemcpy(result.data(), deviceC, size * sizeof(FloatType), cudaMemcpyDeviceToHost);

        cudaFree(deviceA);
        cudaFree(deviceC);
        return std::make_pair(result, elapsedTime * 1e-3);
    }

    template class ImplCublas<float>;
}
