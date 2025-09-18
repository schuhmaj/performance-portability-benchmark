#include "Impl_CudaTensor.cuh"
#include <mma.h>

namespace ppb {

    constexpr unsigned int TILE_SIZE = 16;

    __global__ void matrixMultiplicationTensor(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, const int M, const int N,
                                         const int K) {
        using namespace nvcuda;
        __shared__ half shrA[TILE_SIZE * TILE_SIZE];
        __shared__ half shrB[TILE_SIZE * TILE_SIZE];
        const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int column = blockIdx.y * blockDim.y + threadIdx.y;

        const unsigned int tileRow = blockIdx.x * TILE_SIZE;
        const unsigned int tileCol = blockIdx.y * TILE_SIZE;

        const unsigned int tx = threadIdx.x;
        const unsigned int ty = threadIdx.y;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        const unsigned int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        for (int i = 0; i < numTiles; ++i) {
            const unsigned int offset = tx + TILE_SIZE * ty;
            const unsigned int kA = i * TILE_SIZE + ty;
            const unsigned int kB = i * TILE_SIZE + tx;
            shrA[offset] = __float2half((row < M && kA < K) ? a[row + M * kA] : 0.0f);
            shrB[offset] = __float2half((column < N && kB < K) ? b[column * K + kB] : 0.0f);
            __syncthreads();
            wmma::load_matrix_sync(a_frag, shrA, 16);
            wmma::load_matrix_sync(b_frag, shrB, 16);

            wmma::mma_sync(acc, a_frag, b_frag, acc);
            __syncthreads();
        }
        wmma::store_matrix_sync(&c[tileRow + tileCol * M], acc, M, wmma::mem_col_major);
    }

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplCudaTensor<FloatType>::operator()(const std::vector<FloatType> &a,
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

        const dim3 blockSize = dim3(TILE_SIZE, TILE_SIZE, 1);
        const dim3 gridSize = getIdealGridSize(blockSize, config.m, config.n);

        cudaMallocAsync(&devA, sizeA, stream);
        cudaMallocAsync(&devB, sizeB, stream);
        cudaMallocAsync(&devC, sizeC, stream);
        cudaMemsetAsync(&devC, 0, sizeC, stream);

        cudaMemcpyAsync(devA, a.data(), sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devB, b.data(), sizeB, cudaMemcpyHostToDevice, stream);

        cudaEventRecord(start, stream);
        matrixMultiplicationTensor<<<gridSize, blockSize, 0, stream>>>(devA, devB, devC, config.m, config.n, config.k);
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
    dim3 ImplCudaTensor<FloatType>::getIdealGridSize(const dim3 &blockSize, const int m, const int n) {
        return {ceilDiv<unsigned int>(m, blockSize.x), ceilDiv<unsigned int>(n, blockSize.y), 1};
    }


    /* Explicit Instantiation for float and double */
    template class ImplCudaTensor<float>;
} // namespace ppb
