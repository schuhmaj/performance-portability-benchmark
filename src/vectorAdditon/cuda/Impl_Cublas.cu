#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <utility>

namespace ppb {
    template <typename FloatType>
    struct ImplCublas {
        using float_type = FloatType;

        cublasHandle_t handle;

        ImplCublas()
        : handle {nullptr} {
            cublasCreate(&handle);
        }

        ~ImplCublas() {
            cublasDestroy(handle);
        }

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
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
    };

    template class ImplCublas<float>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplCublas<float>>::benchmark)
    ->Name("VecAdd-Cublas-Float")
    ->RangeMultiplier(10)
    ->Range(1e6, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}