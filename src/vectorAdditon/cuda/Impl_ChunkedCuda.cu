#include "vectorAdditon/cuda/Implementations.cuh"

namespace ppb {

    // Kernel for vector addition
    __global__ void kernel_vector_add(int size, float *__restrict__ a, float *__restrict__ b, float *__restrict__ c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            c[i] = a[i] + b[i];
        }
    }

    template <typename FloatType>
    ImplChunkedCuda<FloatType>::ImplChunkedCuda() :
        stream{} {
        size_t freeMemory, totalMemory;
        cudaMemGetInfo(&freeMemory, &totalMemory);
        size_t usableMemory = static_cast<size_t>(static_cast<double>(freeMemory) * 0.8);
        chunkSize = usableMemory / (3 * NUM_STREAMS * sizeof(FloatType));
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
        }
    }

    template <typename FloatType>
    ImplChunkedCuda<FloatType>::~ImplChunkedCuda() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(stream[i]);
            cudaStreamDestroy(stream[i]);
        }
    }

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplChunkedCuda<FloatType>::operator()(
        const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
        float elapsedTime = 0.0;
        float totalElapsedTime = 0.0;
        cudaEvent_t start[4], stop[4];
        for (int i = 0; i < 4; i++) {
            cudaEventCreate(&start[i]);
            cudaEventCreate(&stop[i]);
        }
        const size_t _size = a.size();
        std::vector<FloatType> result(_size);
        FloatType *deviceA[NUM_STREAMS];
        FloatType *deviceB[NUM_STREAMS];
        FloatType *deviceC[NUM_STREAMS];

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaMallocAsync(&deviceA[i], chunkSize * sizeof(FloatType), stream[i]);
            cudaMallocAsync(&deviceB[i], chunkSize * sizeof(FloatType), stream[i]);
            cudaMallocAsync(&deviceC[i], chunkSize * sizeof(FloatType), stream[i]);
        }

        int minGridSize;
        int blockSize = 256;
        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            kernel_vector_add,
            0,
            _size
            );

        size_t neededChunks = (_size + chunkSize - 1) / chunkSize;
        for (size_t chunk_idx = 0; chunk_idx < neededChunks; ++chunk_idx) {
            int stream_idx = chunk_idx % NUM_STREAMS;
            size_t offset = chunk_idx * chunkSize;
            size_t current_chunk_size = std::min(chunkSize, _size - offset);

            cudaMemcpyAsync(deviceA[stream_idx], &a[offset],
                            current_chunk_size * sizeof(FloatType),
                            cudaMemcpyHostToDevice, stream[stream_idx]);
            cudaMemcpyAsync(deviceB[stream_idx], &b[offset],
                            current_chunk_size * sizeof(FloatType),
                            cudaMemcpyHostToDevice, stream[stream_idx]);

            int gridSize = (current_chunk_size + blockSize - 1) / blockSize;
            if (chunk_idx > NUM_STREAMS) {
                cudaEventSynchronize(stop[stream_idx]);
                cudaEventElapsedTime(&elapsedTime, start[stream_idx], stop[stream_idx]);
                totalElapsedTime += elapsedTime;
            }
            cudaEventRecord(start[stream_idx]);
            kernel_vector_add<<<gridSize, blockSize, 0, stream[stream_idx]>>>(current_chunk_size,
                                                                              deviceA[stream_idx], deviceB[stream_idx],
                                                                              deviceC[stream_idx]);
            cudaEventRecord(stop[stream_idx]);

            // Copy result back
            cudaMemcpyAsync(&result[offset], deviceC[stream_idx],
                            current_chunk_size * sizeof(FloatType),
                            cudaMemcpyDeviceToHost, stream[stream_idx]);
        }

        // Wait for all streams to complete
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(stream[i]);
            cudaFreeAsync(deviceA[i], stream[i]);
            cudaFreeAsync(deviceB[i], stream[i]);
            cudaFreeAsync(deviceC[i], stream[i]);
            cudaEventSynchronize(stop[i]);
            cudaEventElapsedTime(&elapsedTime, start[i], stop[i]);
            totalElapsedTime += elapsedTime;
        }
        return std::make_pair(result, totalElapsedTime * 1e6);
    }

    template class ImplChunkedCuda<float>;

}

// int main(int argc, char** argv) {
//     benchmark::MaybeReenterWithoutASLR(argc, argv);
//     benchmark::Initialize(&argc, argv);
//     benchmark::RunSpecifiedBenchmarks();
//     benchmark::Shutdown();
//
//     // int numBlocks;
//     // int blockSize = 768;
//
//     // // These variables are used to convert occupancy to warps
//     // int device;
//     // cudaDeviceProp prop;
//     // int activeWarps;
//     // int maxWarps;
//
//     // cudaGetDevice(&device);
//     // cudaGetDeviceProperties(&prop, device);
//
//     // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//     //     &numBlocks,
//     //     kernel_vector_add<float>,
//     //     blockSize,
//     //     0);
//
//
//     // activeWarps = numBlocks * blockSize / prop.warpSize;
//     // maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
//
//     // std::cout << "Device: " << prop.name << '\n'
//     //           << "Wrap Size: " << prop.warpSize << '\n'
//     //           << "Active Warps: " << activeWarps << '\n'
//     //           << "Max Warps: " << maxWarps << '\n'
//     //           << "Max Threads Per Block: " << prop.maxThreadsPerBlock << '\n'
//     //           << "Max Threads Per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << '\n'
//     //           << "Max Blocks Per MultiProcessor: " << numBlocks << '\n'
//     //           << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
//
//     // blockSize = 0;
//     // int minGridSize;    // The minimum grid size needed to achieve the
//     //                     // maximum occupancy for a full device
//     //                     // launch
//     // int gridSize;       // The actual grid size needed, based on input
//     //                     // size
//
//     // int arrayCount = 1e6; // Example size, replace with actual size
//     // cudaOccupancyMaxPotentialBlockSize(
//     //     &minGridSize,
//     //     &blockSize,
//     //     (void*)kernel_vector_add<float>,
//     //     0,
//     //     arrayCount
//     // );
//     // gridSize = (arrayCount + blockSize - 1) / blockSize;
//     // std::cout << "Block Size: " << blockSize << '\n'
//     //           << "Min Grid Size: " << minGridSize << '\n'
//     //           << "Grid Size: " << gridSize << '\n'
//     //           << "Occupancy: " << (double)gridSize / maxWarps * 100 << "%" << std::endl;
// }
