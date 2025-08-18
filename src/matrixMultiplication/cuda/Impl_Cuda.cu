#include "Impl_Cuda.cuh"

namespace ppb {

    __global__ void matrixMultiplication(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, const int m, const int n,
                                         const int k) {
        extern __shared__ float shrA[];
        const unsigned int threadsInBlock = blockDim.x * blockDim.y;
        float *__restrict__ shrB = shrA + threadsInBlock;
        const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int column = blockIdx.y * blockDim.y + threadIdx.y;

        const unsigned int tx = threadIdx.x;
        const unsigned int ty = threadIdx.y;
        if (row >= n || column >= m) {
            return;
        }

        float sum = 0.0;
        const unsigned int numTiles = (k + blockDim.x - 1) / blockDim.x;
        #pragma unroll
        for (int i = 0; i < numTiles; ++i) {
            const unsigned int offset = tx + blockDim.x * ty;
            shrA[offset] = a[row + m * (ty + i * blockDim.x)];
            shrB[offset] = b[column * k + (tx + i * blockDim.x)];
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < blockDim.x; ++j) {
                sum += shrA[tx + j * blockDim.x] * shrB[j + ty * blockDim.x];
            }
            __syncthreads();
        }
        c[row + column * m] = sum;
    }

    template <typename FloatType>
    std::vector<FloatType> ImplCuda<FloatType>::operator()(const std::vector<FloatType> &a,
                                                                const std::vector<FloatType> &b,
                                                                const MatrixMultiplicationConfig &config) {

        cudaStream_t stream = cudaStreamPerThread;

        // Allocate device memory
        FloatType *devA, *devB, *devC;
        const size_t sizeA = config.m * config.k * sizeof(FloatType);
        const size_t sizeB = config.k * config.n * sizeof(FloatType);
        const size_t sizeC = config.m * config.n * sizeof(FloatType);

        const dim3 blockSize = getIdealBlockSize(config.m, config.n);
        const dim3 gridSize = getIdealGridSize(blockSize, config.m, config.n);
        const size_t sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(FloatType);

        cudaMallocAsync(&devA, sizeA, stream);
        cudaMallocAsync(&devB, sizeB, stream);
        cudaMallocAsync(&devC, sizeC, stream);
        cudaMemsetAsync(&devC, 0, sizeC, stream);

        cudaMemcpyAsync(devA, a.data(), sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devB, b.data(), sizeB, cudaMemcpyHostToDevice, stream);
        
        matrixMultiplication<<<gridSize, blockSize, sharedMemSize, stream>>>(devA, devB, devC, config.m, config.n, config.k);

        std::vector<FloatType> result(config.m * config.n, 0.0);
        cudaMemcpyAsync(result.data(), devC, sizeC, cudaMemcpyDeviceToHost, stream);

        cudaFreeAsync(devA, stream);
        cudaFreeAsync(devB, stream);
        cudaFreeAsync(devC, stream);
        return result;
    }

    template <typename FloatType>
    dim3 ImplCuda<FloatType>::getIdealBlockSize(const unsigned int m, const unsigned int n) {
        constexpr unsigned int WRAP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;
        const int blockSizeLimit = static_cast<int>(m) * static_cast<int>(n);
        if (blockSizeLimit <= MAX_THREADS) {
            return {m, n, 1};
        }
        int blockSize = 0;
        int minGridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reinterpret_cast<void *>(matrixMultiplication), 0, blockSizeLimit);

        // blockSize is most likely either 768 (32x24) or 1024 (32x32) given the current GPUs
        // Number of Resident Threads varies between 1024, 1536 and 2048; maximum block size is always 1024
        // Hence, it's either 1x1024 per SM, 2x768 per SM or 2x1024 per SM
        return {WRAP_SIZE, blockSize / WRAP_SIZE, 1};
    }

    template <typename FloatType>
    dim3 ImplCuda<FloatType>::getIdealGridSize(const dim3 &blockSize, const int m, const int n) {
        return {(m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y, 1};
    }


    /* Explicit Instantiation for float and double */
    template class ImplCuda<float>;
    //template class ImplCuda<double>;
} // namespace ppb
