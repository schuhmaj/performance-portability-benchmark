#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include "VectorAddition.h"

// Kernel for vector addition using shared memory
template <typename FloatType>
__global__ void kernel_vector_add(int size, const FloatType *__restrict__ a, const FloatType *__restrict__ b,
                                  FloatType *__restrict__ c) {
    extern __shared__ float data[];

    FloatType* sharedA = data;
    FloatType* sharedB = data + blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements from global memory to shared memory
    if (i < size) {
        sharedA[threadIdx.x] = a[i];
        sharedB[threadIdx.x] = b[i];
    }

    // Ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Perform the addition in shared memory
    if (i < size) {
        c[i] = sharedA[threadIdx.x] + sharedB[threadIdx.x];
    }
}

// VectorAddition operator() implementation
template <typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    // Step 1: Create the device vectors and allocate memory
    FloatType *deviceA;
    FloatType *deviceB;
    FloatType *deviceC;
    const size_t dataSize = _inA.size() * sizeof(FloatType);

    cudaMalloc(&deviceA, dataSize);
    cudaMalloc(&deviceB, dataSize);
    cudaMalloc(&deviceC, dataSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Step 2: Transfer the input data to the device asynchronously
    cudaMemcpyAsync(deviceA, _inA.data(), dataSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceB, _inB.data(), dataSize, cudaMemcpyHostToDevice, stream);

    // Step 3: Execute the kernel
    const dim3 threadsPerBlock(1024);
    const dim3 numBlocks((_inA.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemSize = 2 * threadsPerBlock.x * sizeof(FloatType); // Allocate shared memory for a and b

    kernel_vector_add<<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(_inA.size(), deviceA, deviceB, deviceC);

    // Synchronize to check for kernel execution errors
    cudaStreamSynchronize(stream);

    // Step 4: Copy the result back to the host asynchronously and free the memory
    cudaMemcpyAsync(_outC.data(), deviceC, dataSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    checkValidity();
    return _outC;
}

// Explicit instantiation and benchmarking setup
template __global__ void kernel_vector_add<float>(int, const float *__restrict__, const float *__restrict__,
                                                  float *__restrict__);
template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)
    ->Name("VecAdd-Cuda3-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
