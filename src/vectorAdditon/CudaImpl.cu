#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <iostream>

// Kernel for vector addition
template<typename FloatType>
__global__ void kernel_vector_add(int size, FloatType* a, FloatType* b, FloatType* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// VectorAddition operator() implementation
template<typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    // Step 1: Create the device vectors and allocate memory
    FloatType *deviceA, *deviceB, *deviceC;
    const size_t dataSize = _inA.size() * sizeof(FloatType);

    cudaMalloc(&deviceA, dataSize);
    cudaMalloc(&deviceB, dataSize);
    cudaMalloc(&deviceC, dataSize);

    // Step 2: Transfer the input data to the device
    cudaMemcpy(deviceA, _inA.data(), dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, _inB.data(), dataSize, cudaMemcpyHostToDevice);

    // Step 3: Execute the kernel
    const dim3 threadsPerBlock(256);
    const dim3 numBlocks((_inA.size() + threadsPerBlock.x - 1) / threadsPerBlock.x); // Adjust numBlocks to cover all elements
    kernel_vector_add<<<numBlocks, threadsPerBlock>>>(_inA.size(), deviceA, deviceB, deviceC);

    // Synchronize to check for kernel execution errors
    cudaDeviceSynchronize();

    // Step 4: Copy the result back to the host and free the memory
    cudaMemcpy(_outC.data(), deviceC, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    checkValidity();
    return _outC;
}

// Explicit instantiation and benchmarking setup
template __global__ void kernel_vector_add<float>(int, float*, float*, float*);
template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)->Name("Vector Addition Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

template __global__ void kernel_vector_add<double>(int, double*, double*, double*);
template std::vector<double> VectorAddition<double>::operator()();
BENCHMARK(VectorAddition<double>::vectorAdditionBenchmark)->Name("Vector Addition Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}