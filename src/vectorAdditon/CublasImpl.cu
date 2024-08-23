#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

template<typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    // Step 1: Create the device vectors and allocate memory
    FloatType *deviceA, *deviceB, *deviceC;
    const size_t dataSize = _inA.size() * sizeof(FloatType);
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc(&deviceA, dataSize);
    cudaMalloc(&deviceB, dataSize);
    cudaMalloc(&deviceC, dataSize);

    // Step 2: Transfer the input data to the device
    cudaMemcpy(deviceA, _inA.data(), dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, _inB.data(), dataSize, cudaMemcpyHostToDevice);

    // Step 3: Execute the cuBLAS axpy
    const FloatType alpha = 1.0; // y = alpha * x + y
    if constexpr (std::is_same<FloatType, float>::value) {
        cublasSaxpy(handle, _inA.size(), &alpha, deviceA, 1, deviceB, 1);
    } else if constexpr (std::is_same<FloatType, double>::value) {
        cublasDaxpy(handle, _inA.size(), &alpha, deviceA, 1, deviceB, 1);
    }

    // Step 4: Copy the result back to the host and free the memory
    cudaMemcpy(_outC.data(), deviceB, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    cublasDestroy(handle);

    checkValidity();
    return _outC;
}

// Explicit instantiation and benchmarking setup
template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)->Name("VecAdd-Cublas-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

template std::vector<double> VectorAddition<double>::operator()();
BENCHMARK(VectorAddition<double>::vectorAdditionBenchmark)->Name("VecAdd-Cublas-Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}