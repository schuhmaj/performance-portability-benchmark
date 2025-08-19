#include "Impl_CudaTensor.cuh"
#include <mma.h>

namespace ppb {

    constexpr unsigned int TILE_SIZE = 16;

    __global__ void matrixMultiplicationTensor(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, const int M, const int N,
                                         const int K) {
        __shared__ float shrA[TILE_SIZE * TILE_SIZE];
        __shared__ float shrB[TILE_SIZE * TILE_SIZE];
        const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int column = blockIdx.y * blockDim.y + threadIdx.y;

        const unsigned int tx = threadIdx.x;
        const unsigned int ty = threadIdx.y;

        float sum = 0.0;
        const unsigned int numTiles = (K + blockDim.x - 1) / blockDim.x;
        #pragma unroll
        for (int i = 0; i < numTiles; ++i) {
            const unsigned int offset = tx + blockDim.x * ty;
            const unsigned int kA = i * blockDim.x + ty;
            const unsigned int kB = i * blockDim.x + tx;
            shrA[offset] = (row < M && kA < K) ? a[row + M * kA] : 0.0f;
            shrB[offset] = (column < N && kB < K) ? b[column * K + kB] : 0.0f;
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < blockDim.x; ++j) {
                sum += shrA[tx + j * blockDim.x] * shrB[j + ty * blockDim.x];
            }
            __syncthreads();
        }
        if (row < N && column < M) {
            c[row + column * M] = sum;
        }

    }

    template <typename FloatType>
    std::vector<FloatType> ImplCudaTensor<FloatType>::operator()(const std::vector<FloatType> &a,
                                                                const std::vector<FloatType> &b,
                                                                const MatrixMultiplicationConfig &config) {

        cudaStream_t stream = cudaStreamPerThread;

        // Allocate device memory
        FloatType *devA, *devB, *devC;
        const size_t sizeA = config.m * config.k * sizeof(FloatType);
        const size_t sizeB = config.k * config.n * sizeof(FloatType);
        const size_t sizeC = config.m * config.n * sizeof(FloatType);

        const dim3 blockSize = dim3(TILE_SIZE, TILE_SIZE, 1);
        const dim3 gridSize = getIdealGridSize(blockSize, config.m, config.n);

        cudaMallocAsync(&devA, sizeA, stream);
        cudaMallocAsync(&devB, sizeB, stream);
        cudaMallocAsync(&devC, sizeC, stream);
        cudaMemsetAsync(&devC, 0, sizeC, stream);

        cudaMemcpyAsync(devA, a.data(), sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devB, b.data(), sizeB, cudaMemcpyHostToDevice, stream);

        matrixMultiplicationTensor<<<gridSize, blockSize, 0, stream>>>(devA, devB, devC, config.m, config.n, config.k);

        std::vector<FloatType> result(config.m * config.n, 0.0);
        cudaMemcpyAsync(result.data(), devC, sizeC, cudaMemcpyDeviceToHost, stream);

        cudaFreeAsync(devA, stream);
        cudaFreeAsync(devB, stream);
        cudaFreeAsync(devC, stream);
        return result;
    }

    template <typename FloatType>
    dim3 ImplCudaTensor<FloatType>::getIdealGridSize(const dim3 &blockSize, const int m, const int n) {
        return {ceilDiv<unsigned int>(m, blockSize.x), ceilDiv<unsigned int>(n, blockSize.y), 1};
    }


    /* Explicit Instantiation for float and double */
    template class ImplCudaTensor<float>;
} // namespace ppb
