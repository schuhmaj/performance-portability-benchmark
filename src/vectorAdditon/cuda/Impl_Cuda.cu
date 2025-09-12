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

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
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

            kernel_vector_add<<<gridSize, blockSize, 0, stream>>>(size, deviceA, deviceC);

            std::vector<FloatType> result(size);
            cudaMemcpyAsync(result.data(), deviceC, size * sizeof(FloatType), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            cudaFree(deviceA);
            cudaFree(deviceC);
            return result;
        }
    };

    template class ImplCuda<float>;

}

BENCHMARK(ppb::VectorAddition<ppb::ImplCuda<float>>::benchmark)->Name("VecAdd-Cuda-Float")->RangeMultiplier(10)->Range(1e6, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}