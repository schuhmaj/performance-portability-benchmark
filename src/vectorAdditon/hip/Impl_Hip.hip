#include "vectorAdditon/VectorAddition.h"
#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>


namespace ppb {
    // Kernel for vector addition
    __global__ void kernel_vector_add(int size, float *__restrict__ a, float *__restrict__ c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            c[i] = a[i] + c[i];
        }
    }

    template <typename FloatType>
    struct ImplHip {
        using float_type = FloatType;
        hipStream_t stream;
        ImplHip();
        ~ImplHip();
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a,
                                                             const std::vector<FloatType> &b);
    };

    template <typename FloatType>
    ImplHip<FloatType>::ImplHip() :
        stream{} {
        hipStreamCreate(&stream);
    }

    template <typename FloatType>
    ImplHip<FloatType>::~ImplHip() {
        hipStreamDestroy(stream);
    }

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplHip<FloatType>::operator()(
        const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
        float elapsedTime;
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        const size_t size = a.size();
        FloatType *deviceA;
        FloatType *deviceC;

        hipMallocAsync(reinterpret_cast<void**>(&deviceA), size * sizeof(FloatType), stream);
        hipMallocAsync(reinterpret_cast<void**>(&deviceC), size * sizeof(FloatType), stream);
        hipMemcpyAsync(deviceA, a.data(), size * sizeof(FloatType), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(deviceC, b.data(), size * sizeof(FloatType), hipMemcpyHostToDevice, stream);

        int minGridSize;
        int blockSize = 256;
        hipOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            kernel_vector_add,
            0,
            size
            );
        int gridSize = (size + blockSize - 1) / blockSize;

        hipEventRecord(start, stream);
        // NVMON_MARKER_START("triad");
        kernel_vector_add<<<gridSize, blockSize, 0, stream>>>(size, deviceA, deviceC);
        // NVMON_MARKER_STOP("triad");
        hipEventRecord(stop, stream);

        std::vector<FloatType> result(size);
        hipMemcpyAsync(result.data(), deviceC, size * sizeof(FloatType), hipMemcpyDeviceToHost, stream);
        hipFreeAsync(deviceA, stream);
        hipFreeAsync(deviceC, stream);
        hipStreamSynchronize(stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);
        return std::make_pair(result, elapsedTime * 1e6);
    };

    template class ImplHip<float>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplHip<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd-Naive")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
    #ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
    #endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("Hip");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}