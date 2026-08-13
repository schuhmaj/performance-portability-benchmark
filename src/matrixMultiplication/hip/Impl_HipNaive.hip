#include "Impl_HipNaive.h"

namespace ppb {

    template<typename FloatType>
    __global__ void matrixMultiplicationHip(const FloatType *a, const FloatType *b, FloatType *c,
                                            const int M, const int N, const int K) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= N || j >= M) {
            return;
        }
        FloatType sum = 0.0;
        for (unsigned int entry = 0; entry < K; ++entry) {
            sum += a[i + entry * M] * b[entry + j * K];
        }
        c[i + j * M] = sum;
    }

    template __global__ void matrixMultiplicationHip<float>(const float*, const float*, float*, const int, const int, const int);

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplHipNaive<FloatType>::operator()(const std::vector<FloatType> &a,
                                                                                   const std::vector<FloatType> &b,
                                                                                   const MatrixMultiplicationConfig &config) {
        float elapsedTime;
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipStream_t stream = hipStreamPerThread;

        FloatType *devA, *devB, *devC;
        const size_t sizeA = config.m * config.k * sizeof(FloatType);
        const size_t sizeB = config.k * config.n * sizeof(FloatType);
        const size_t sizeC = config.m * config.n * sizeof(FloatType);

        const dim3 blockSize = getIdealBlockSize(config.m, config.n);
        const dim3 gridSize = getIdealGridSize(blockSize, config.m, config.n);

        hipMallocAsync(reinterpret_cast<void**>(&devA), sizeA, stream);
        hipMallocAsync(reinterpret_cast<void**>(&devB), sizeB, stream);
        hipMallocAsync(reinterpret_cast<void**>(&devC), sizeC, stream);

        hipMemcpyAsync(devA, a.data(), sizeA, hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(devB, b.data(), sizeB, hipMemcpyHostToDevice, stream);
        hipEventRecord(start, stream);
        matrixMultiplicationHip<<<gridSize, blockSize, 0, stream>>>(devA, devB, devC, config.m, config.n, config.k);
        hipEventRecord(stop, stream);
        std::vector<FloatType> result(config.m * config.n, 0.0);
        hipMemcpyAsync(result.data(), devC, sizeC, hipMemcpyDeviceToHost, stream);

        hipFreeAsync(devA, stream);
        hipFreeAsync(devB, stream);
        hipFreeAsync(devC, stream);
        hipStreamSynchronize(stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);
        return std::make_pair(result, elapsedTime * 1e6);
    }

    template <typename FloatType>
    dim3 ImplHipNaive<FloatType>::getIdealBlockSize(const unsigned int m, const unsigned int n) {
        constexpr unsigned int WARP_SIZE = 32;
        const int blockSizeLimit = static_cast<int>(m) * static_cast<int>(n);
        int blockSize = 0;
        int minGridSize = 0;
        hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                          reinterpret_cast<const void*>(matrixMultiplicationHip<float>),
                                          0, blockSizeLimit);
        if (blockSize == blockSizeLimit) {
            return {m, n, 1};
        }
        return {WARP_SIZE, static_cast<unsigned int>(blockSize) / WARP_SIZE, 1};
    }

    template <typename FloatType>
    dim3 ImplHipNaive<FloatType>::getIdealGridSize(const dim3 &blockSize, const int m, const int n) {
        return {(static_cast<unsigned int>(m) + blockSize.x - 1) / blockSize.x,
                (static_cast<unsigned int>(n) + blockSize.y - 1) / blockSize.y, 1};
    }

    template class ImplHipNaive<float>;
    template class ImplHipNaive<double>;

} // namespace ppb
