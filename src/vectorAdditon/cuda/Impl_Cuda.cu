#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <iostream>

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

    FloatType* deviceA;
    FloatType* deviceB;
    FloatType* deviceC;
    cudaStream_t stream;

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b)
    : deviceA{nullptr},
      deviceB {nullptr},
      deviceC {nullptr},
      stream{} {
        cudaStreamCreate(&stream);
        cudaMallocAsync(&deviceA, size * sizeof(FloatType), stream);
        cudaMallocAsync(&deviceB, size * sizeof(FloatType), stream);
        cudaMallocAsync(&deviceC, size * sizeof(FloatType), stream);

        cudaMemcpyAsync(deviceA, a.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(deviceB, b.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice, stream);
    }

    ~impl() {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
    }
};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

// Kernel for vector addition
template<typename FloatType>
__global__ void kernel_vector_add(int size, FloatType* __restrict__ a, FloatType* __restrict__ b, FloatType* __restrict__ c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// VectorAddition operator() implementation
template<typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    std::vector<FloatType> result(_size);

    int minGridSize;
    int blockSize = 256;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)kernel_vector_add<FloatType>,
        0,
        _size
    );
    int gridSize = (_size + blockSize - 1) / blockSize;

    kernel_vector_add<<<gridSize, blockSize, 0, _impl->stream>>>(_size, _impl->deviceA, _impl->deviceB, _impl->deviceC);

    cudaMemcpyAsync(result.data(), _impl->deviceC, _size * sizeof(FloatType), cudaMemcpyDeviceToHost, _impl->stream);
    cudaStreamSynchronize(_impl->stream);
    return result;
}

// Explicit instantiation and benchmarking setup
template __global__ void kernel_vector_add<float>(int, float*, float*, float*);
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-Cuda-Float")->RangeMultiplier(10)->Range(1e6, 1e9)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}