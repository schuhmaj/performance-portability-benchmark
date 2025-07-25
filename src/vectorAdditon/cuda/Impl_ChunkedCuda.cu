#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include <iostream>

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

    static constexpr size_t NUM_STREAMS = 4;
    size_t chunkSize;

    FloatType* deviceA[NUM_STREAMS];
    FloatType* deviceB[NUM_STREAMS];
    FloatType* deviceC[NUM_STREAMS];
    cudaStream_t stream[NUM_STREAMS];

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b)
    : deviceA{nullptr},
      deviceB {nullptr},
      deviceC {nullptr},
      stream{} {
        size_t freeMemory, totalMemory;
        cudaMemGetInfo(&freeMemory, &totalMemory);
        size_t usableMemory = static_cast<size_t>(static_cast<double>(freeMemory) * 0.8);
        chunkSize = usableMemory / (3 * NUM_STREAMS * sizeof(FloatType));

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
            cudaMallocAsync(&deviceA[i], chunkSize * sizeof(FloatType), stream[i]);
            cudaMallocAsync(&deviceB[i], chunkSize * sizeof(FloatType), stream[i]);
            cudaMallocAsync(&deviceC[i], chunkSize * sizeof(FloatType), stream[i]);
        }
    }

    ~impl() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaFreeAsync(deviceA[i], stream[i]);
            cudaFreeAsync(deviceB[i], stream[i]);
            cudaFreeAsync(deviceC[i], stream[i]);
            cudaStreamSynchronize(stream[i]);
            cudaStreamDestroy(stream[i]);
        }
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

    size_t neededChunks = (_size + _impl->chunkSize - 1) / _impl->chunkSize;

    for (size_t chunk_idx = 0; chunk_idx < neededChunks; ++chunk_idx) {
        int stream_idx = chunk_idx % _impl->NUM_STREAMS;
        size_t offset = chunk_idx * _impl->chunkSize;
        size_t current_chunk_size = std::min(_impl->chunkSize, _size - offset);

        cudaMemcpyAsync(_impl->deviceA[stream_idx], &_inA[offset],
                       current_chunk_size * sizeof(FloatType),
                       cudaMemcpyHostToDevice, _impl->stream[stream_idx]);
        cudaMemcpyAsync(_impl->deviceB[stream_idx],  &_inB[offset],
                       current_chunk_size * sizeof(FloatType),
                       cudaMemcpyHostToDevice, _impl->stream[stream_idx]);

        int gridSize = (current_chunk_size + blockSize - 1) / blockSize;
        kernel_vector_add<<<gridSize, blockSize, 0, _impl->stream[stream_idx]>>>(current_chunk_size,
            _impl->deviceA[stream_idx], _impl->deviceB[stream_idx],
            _impl->deviceC[stream_idx]);

        // Copy result back
        cudaMemcpyAsync(&result[offset], _impl->deviceC[stream_idx],
                       current_chunk_size * sizeof(FloatType),
                       cudaMemcpyDeviceToHost, _impl->stream[stream_idx]);
    }

    // Wait for all streams to complete
    for (int i = 0; i < _impl->NUM_STREAMS; ++i) {
        cudaStreamSynchronize(_impl->stream[i]);
    }
    return result;
}

// Explicit instantiation and benchmarking setup
template __global__ void kernel_vector_add<float>(int, float*, float*, float*);
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-Cuda-Float")->RangeMultiplier(10)->Range(1e6, 1e9)->Complexity();

// template __global__ void kernel_vector_add<double>(int, double*, double*, double*);
// template std::vector<double> ppb::VectorAddition<double>::operator()();
// BENCHMARK(ppb::VectorAddition<double>::benchmark)->Name("VecAdd-Cuda-Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    // int numBlocks;
    // int blockSize = 768;

    // // These variables are used to convert occupancy to warps
    // int device;
    // cudaDeviceProp prop;
    // int activeWarps;
    // int maxWarps;

    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);

    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocks,
    //     kernel_vector_add<float>,
    //     blockSize,
    //     0);


    // activeWarps = numBlocks * blockSize / prop.warpSize;
    // maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    // std::cout << "Device: " << prop.name << '\n'
    //           << "Wrap Size: " << prop.warpSize << '\n'
    //           << "Active Warps: " << activeWarps << '\n'
    //           << "Max Warps: " << maxWarps << '\n'
    //           << "Max Threads Per Block: " << prop.maxThreadsPerBlock << '\n'
    //           << "Max Threads Per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << '\n'
    //           << "Max Blocks Per MultiProcessor: " << numBlocks << '\n'
    //           << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

    // blockSize = 0;
    // int minGridSize;    // The minimum grid size needed to achieve the
    //                     // maximum occupancy for a full device
    //                     // launch
    // int gridSize;       // The actual grid size needed, based on input
    //                     // size

    // int arrayCount = 1e6; // Example size, replace with actual size
    // cudaOccupancyMaxPotentialBlockSize(
    //     &minGridSize,
    //     &blockSize,
    //     (void*)kernel_vector_add<float>,
    //     0,
    //     arrayCount
    // );
    // gridSize = (arrayCount + blockSize - 1) / blockSize;
    // std::cout << "Block Size: " << blockSize << '\n'
    //           << "Min Grid Size: " << minGridSize << '\n'
    //           << "Grid Size: " << gridSize << '\n'
    //           << "Occupancy: " << (double)gridSize / maxWarps * 100 << "%" << std::endl;
}