#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

    FloatType* deviceA;
    FloatType* deviceB;
    FloatType* deviceC;
    cudaStream_t stream;
    cublasHandle_t handle;

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b)
    : deviceA{nullptr},
      deviceB {nullptr},
      deviceC{nullptr},
      stream{nullptr} {
        cudaStreamCreate(&stream);
        cudaMallocAsync(&deviceA, size * sizeof(FloatType), stream);
        cudaMallocAsync(&deviceB, size * sizeof(FloatType), stream);
        cudaMallocAsync(&deviceC, size * sizeof(FloatType), stream);
        cudaMemcpyAsync(deviceA, a.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(deviceB, b.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice, stream);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }

    ~impl() {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        cublasDestroy(handle);
    }
};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

template<typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    std::vector<FloatType> result(_size);
    cudaMemcpyAsync(_impl->deviceC, _impl->deviceB, _size * sizeof(FloatType), cudaMemcpyDeviceToDevice, _impl->stream);

    constexpr FloatType alpha = 1.0;
    if constexpr (std::is_same_v<FloatType, float>) {
        cublasSaxpy(_impl->handle, _size, &alpha, _impl->deviceA, 1, _impl->deviceC, 1);
    } else if constexpr (std::is_same_v<FloatType, double>) {
        cublasDaxpy(_impl->handle, _size, &alpha, _impl->deviceA, 1, _impl->deviceC, 1);
    }

    cudaMemcpyAsync(result.data(), _impl->deviceC, _size * sizeof(FloatType), cudaMemcpyDeviceToHost, _impl->stream);
    cudaStreamSynchronize(_impl->stream);
    return result;
}

// Explicit instantiation and benchmarking setup
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-Cublas-Float")->RangeMultiplier(10)->Range(1e6, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}