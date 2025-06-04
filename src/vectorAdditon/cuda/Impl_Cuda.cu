#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <iostream>

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

    FloatType* deviceA;
    FloatType* deviceB;

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b)
    : deviceA{nullptr},
      deviceB {nullptr} {
        cudaMalloc(&deviceA, size * sizeof(FloatType));
        cudaMalloc(&deviceB, size * sizeof(FloatType));
        cudaMemcpy(deviceA, a.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, b.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice);
    }

    ~impl() {
        cudaFree(deviceA);
        cudaFree(deviceB);
    }
};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

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
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    std::vector<FloatType> result(_size);
    FloatType* resultBuffer;
    cudaMalloc(&resultBuffer, _size * sizeof(FloatType));


    const dim3 threadsPerBlock(1024);
    const dim3 numBlocks((_inA.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
    kernel_vector_add<<<numBlocks, threadsPerBlock>>>(_size, _impl->deviceA, _impl->deviceB, resultBuffer);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), resultBuffer, _size * sizeof(FloatType), cudaMemcpyDeviceToHost);
    cudaFree(resultBuffer);
    return result;
}

// Explicit instantiation and benchmarking setup
template __global__ void kernel_vector_add<float>(int, float*, float*, float*);
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-Cuda-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

template __global__ void kernel_vector_add<double>(int, double*, double*, double*);
template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)->Name("VecAdd-Cuda-Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}