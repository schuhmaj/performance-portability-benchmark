#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <iostream>


namespace ppb {
    // Kernel for vector addition
    __global__ void kernel_vector_add(int size, float* __restrict__ a, float* __restrict__ c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            c[i] = a[i] + c[i];
        }
    }

    template <typename FloatType>
    struct ImplCuda {
        using float_type = FloatType;
        cudaStream_t stream;

        ImplCuda()
        :stream{} {
            cudaStreamCreate(&stream);
        }

        ~ImplCuda() {
            cudaStreamDestroy(stream);
        }

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            float elapsedTime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            const size_t size = a.size();
            FloatType* deviceA;
            FloatType* deviceC;

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
            kernel_vector_add<<<gridSize, blockSize, 0, stream>>>(size, deviceA, deviceC);
            cudaEventRecord(stop, stream);

            std::vector<FloatType> result(size);
            cudaMemcpyAsync(result.data(), deviceC, size * sizeof(FloatType), cudaMemcpyDeviceToHost, stream);
            cudaFreeAsync(deviceA, stream);
            cudaFreeAsync(deviceC, stream);
            cudaStreamSynchronize(stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            return std::make_pair(result, elapsedTime * 1e-3);
        }
    };

    template class ImplCuda<float>;

}

BENCHMARK(ppb::VectorAddition<ppb::ImplCuda<float>>::benchmark)
    ->Name("VecAdd-Cuda-Float")
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