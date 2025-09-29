#include "Impl_CudaNaive.cuh"

namespace ppb {

    template<typename FloatType>
    __global__ void matrixMultiplication(const FloatType *a, const FloatType *b, FloatType *c, const int M, const int N,
                                         const int K) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= N || j >= M) {
            return;
        }
        FloatType sum = 0.0;
        for (unsigned int entry = 0; entry < K; ++entry) {
            sum += a[i + entry * M] * b[entry + j * K];
        }
        c[i + j * M] =sum;
    }

    template __global__ void matrixMultiplication<float>(const float*, const float*, float*, const int, const int, const int);

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplCudaNaive<FloatType>::operator()(const std::vector<FloatType> &a,
        const std::vector<FloatType> &b,
        const MatrixMultiplicationConfig &config) {
        float elapsedTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaStream_t stream = cudaStreamPerThread;


        // Allocate device memory
        FloatType *devA, *devB, *devC;
        const size_t sizeA = config.m * config.k * sizeof(FloatType);
        const size_t sizeB = config.k * config.n * sizeof(FloatType);
        const size_t sizeC = config.m * config.n * sizeof(FloatType);

        const dim3 blockSize = getIdealBlockSize(config.m, config.n);
        const dim3 gridSize = getIdealGridSize(blockSize, config.m, config.n);

        cudaMallocAsync(&devA, sizeA, stream);
        cudaMallocAsync(&devB, sizeB, stream);
        cudaMallocAsync(&devC, sizeC, stream);

        cudaMemcpyAsync(devA, a.data(), sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devB, b.data(), sizeB, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(start, stream);
        matrixMultiplication<<<gridSize, blockSize, 0, stream>>>(devA, devB, devC, config.m, config.n, config.k);
        cudaEventRecord(stop, stream);
        std::vector<FloatType> result(config.m * config.n, 0.0);
        cudaMemcpyAsync(result.data(), devC, sizeC, cudaMemcpyDeviceToHost, stream);

        cudaFreeAsync(devA, stream);
        cudaFreeAsync(devB, stream);
        cudaFreeAsync(devC, stream);
        cudaStreamSynchronize(stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        return std::make_pair(result, elapsedTime * 1e-3);
    }

    template <typename FloatType>
    dim3 ImplCudaNaive<FloatType>::getIdealBlockSize(const unsigned int m, const unsigned int n) {
        constexpr unsigned int WRAP_SIZE = 32;
        const int blockSizeLimit = static_cast<int>(m) * static_cast<int>(n);
        int blockSize = 0;
        int minGridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reinterpret_cast<void *>(matrixMultiplication<float>), 0, blockSizeLimit);
        if (blockSize == blockSizeLimit) {
            return {m, n, 1};
        }
        // blockSize is most likely either 768 (32x24) or 1024 (32x32) given the current GPUs
        // Number of Resident Threads varies between 1024, 1536 and 2048; maximum block size is always 1024
        // Hence, it's either 1x1024 per SM, 2x768 per SM or 2x1024 per SM
        return {WRAP_SIZE, blockSize / WRAP_SIZE, 1};
    }

    template <typename FloatType>
    dim3 ImplCudaNaive<FloatType>::getIdealGridSize(const dim3 &blockSize, const int m, const int n) {
        return {(m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y, 1};
    }


    /* Explicit Instantiation for float and double */
    template class ImplCudaNaive<float>;
    template class ImplCudaNaive<double>;
} // namespace ppb
