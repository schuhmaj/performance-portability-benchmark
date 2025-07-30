#include "Impl_Cuda.cuh"

namespace ppb {

    template<typename FloatType>
    __global__ void matrixMultiplication(const FloatType *__restrict__ a, const FloatType* __restrict__ b, FloatType* __restrict__ c, const int m, const int n,
                                         const int l) {
        extern __shared__ float bShared[];
        const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int column = blockIdx.y * blockDim.y + threadIdx.y;
        if (row >= n || column >= m) {
            return;
        }
        // Assign b into bShared; workload every thread: l / (blockDim.x * blockDim.y)
        const unsigned int totalThreadsInBlock = blockDim.x * blockDim.y;
        const unsigned int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
        const unsigned int elementsPerThread = (l + totalThreadsInBlock - 1) / totalThreadsInBlock;

        for (unsigned int i = 0; i < elementsPerThread; ++i) {
            const unsigned int globalIdx = threadIdInBlock * elementsPerThread + i;
            if (globalIdx < l) {
                bShared[globalIdx] = b[globalIdx + column * l];
            }
        }
        __syncthreads();

        FloatType sum = 0.0;
        for (unsigned int i = 0; i < l; ++i) {
            sum += a[row + i * m] * bShared[i];
        }
        c[row + column * m] += sum;
    }

    template __global__ void matrixMultiplication<float>(const float*, const float*, float*, const int, const int, const int);

    template <typename FloatType>
    std::vector<FloatType> ImplCuda<FloatType>::operator()(const std::vector<FloatType> &a,
                                                                const std::vector<FloatType> &b,
                                                                const MatrixMultiplicationConfig &config) {

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Allocate device memory
        FloatType *devA, *devB, *devC;
        const size_t sizeA = config.m * config.l * sizeof(FloatType);
        const size_t sizeB = config.l * config.n * sizeof(FloatType);
        const size_t sizeC = config.m * config.n * sizeof(FloatType);

        const dim3 blockSize = getIdealBlockSize(config.m, config.n);
        const dim3 gridSize = getIdealGridSize(blockSize, config.m, config.n);

        cudaMallocAsync(&devA, sizeA, stream);
        cudaMallocAsync(&devB, sizeB, stream);
        cudaMallocAsync(&devC, sizeC, stream);
        cudaMemsetAsync(&devC, 0, sizeC, stream);

        cudaMemcpyAsync(devA, a.data(), sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devB, b.data(), sizeB, cudaMemcpyHostToDevice, stream);

        matrixMultiplication<<<gridSize, blockSize, config.l * sizeof(FloatType), stream>>>(devA, devB, devC, config.m, config.n, config.l);

        std::vector<FloatType> result(config.m * config.n, 0.0);
        cudaMemcpyAsync(result.data(), devC, sizeC, cudaMemcpyDeviceToHost, stream);

        cudaFreeAsync(devA, stream);
        cudaFreeAsync(devB, stream);
        cudaFreeAsync(devC, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return result;
    }

    template <typename FloatType>
    dim3 ImplCuda<FloatType>::getIdealBlockSize(const unsigned int m, const unsigned int n) {
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
    dim3 ImplCuda<FloatType>::getIdealGridSize(const dim3 &blockSize, const int m, const int n) {
        return {(m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y, 1};
    }


    /* Explicit Instantiation for float and double */
    template class ImplCuda<float>;
    template class ImplCuda<double>;
} // namespace ppb
