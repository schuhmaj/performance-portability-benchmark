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
    size_t dataSize = _inA.size() * sizeof(FloatType);

    cudaError_t err;
    err = cudaMalloc(&deviceA, dataSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for deviceA: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&deviceB, dataSize);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        throw std::runtime_error("Failed to allocate device memory for deviceB: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&deviceC, dataSize);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        throw std::runtime_error("Failed to allocate device memory for deviceC: " + std::string(cudaGetErrorString(err)));
    }

    // Step 2: Transfer the input data to the device
    err = cudaMemcpy(deviceA, _inA.data(), dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        throw std::runtime_error("Failed to copy _inA to device: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(deviceB, _inB.data(), dataSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        throw std::runtime_error("Failed to copy _inB to device: " + std::string(cudaGetErrorString(err)));
    }

    // Step 3: Execute the kernel
    const dim3 threadsPerBlock(256);
    const dim3 numBlocks((_inA.size() + threadsPerBlock.x - 1) / threadsPerBlock.x); // Adjust numBlocks to cover all elements
    kernel_vector_add<<<numBlocks, threadsPerBlock>>>(_inA.size(), deviceA, deviceB, deviceC);

    // Synchronize to check for kernel execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        throw std::runtime_error("Kernel execution failed: " + std::string(cudaGetErrorString(err)));
    }

    // Step 4: Copy the result back to the host and free the memory
    err = cudaMemcpy(_outC.data(), deviceC, dataSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        throw std::runtime_error("Failed to copy deviceC to host: " + std::string(cudaGetErrorString(err)));
    }

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