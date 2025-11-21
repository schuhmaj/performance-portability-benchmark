#include "vectorAdditon/cuda/Implementations.cuh"

#include "vectorAdditon/VectorAddition.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <likwid-marker.h>


namespace ppb {
    // Kernel for vector addition
    __global__ void kernel_vector_add(int size, float *__restrict__ a, float *__restrict__ c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            c[i] = a[i] + c[i];
        }
    }

    template <typename FloatType>
    ImplCuda<FloatType>::ImplCuda() :
        stream{} {
        cudaStreamCreate(&stream);
    }

    template <typename FloatType>
    ImplCuda<FloatType>::~ImplCuda() {
        cudaStreamDestroy(stream);
    }

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplCuda<FloatType>::operator()(
        const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        const size_t size = a.size();
        FloatType *deviceA;
        FloatType *deviceC;

        cudaMallocAsync(&deviceA, size * sizeof(FloatType), stream);
        cudaMallocAsync(&deviceC, size * sizeof(FloatType), stream);
        cudaMemcpyAsync(deviceA, a.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(deviceC, b.data(), size * sizeof(FloatType), cudaMemcpyHostToDevice, stream);

        int minGridSize;
        int blockSize = 256;
        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            kernel_vector_add,
            0,
            size
            );
        int gridSize = (size + blockSize - 1) / blockSize;

        cudaEventRecord(start, stream);
        NVMON_MARKER_START("triad");
        kernel_vector_add<<<gridSize, blockSize, 0, stream>>>(size, deviceA, deviceC);
        NVMON_MARKER_STOP("triad");
        cudaEventRecord(stop, stream);

        std::vector<FloatType> result(size);
        cudaMemcpyAsync(result.data(), deviceC, size * sizeof(FloatType), cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(deviceA, stream);
        cudaFreeAsync(deviceC, stream);
        cudaStreamSynchronize(stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        return std::make_pair(result, elapsedTime * 1e6);
    };

    template class ImplCuda<float>;
}
