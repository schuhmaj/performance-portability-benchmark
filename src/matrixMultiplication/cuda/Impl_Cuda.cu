#include "Impl_Cuda.cuh"

namespace ppb {

    template<typename FloatType>
    __global__ void matrixMultiplication(const FloatType *a, const FloatType *b, FloatType *c, const int m, const int n,
                                         const int l) {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= n || j >= m) {
            return;
        }
        FloatType sum = 0.0;
        for (unsigned int entry = 0; entry < l; ++entry) {
            sum += a[i + entry * m] * b[entry + j * l];
        }
        c[i + j * m] =sum;
    }

    template __global__ void matrixMultiplication<float>(const float*, const float*, float*, const int, const int, const int);

    template <typename FloatType>
    std::vector<FloatType> ImplCuda<FloatType>::operator()(const std::vector<FloatType> &a,
                                                                const std::vector<FloatType> &b,
                                                                const MatrixMultiplicationConfig &config) {

        // Allocate device memory
        FloatType *devA, *devB, *devC;
        const size_t sizeA = config.m * config.l * sizeof(FloatType);
        const size_t sizeB = config.l * config.n * sizeof(FloatType);
        const size_t sizeC = config.m * config.n * sizeof(FloatType);

        cudaMalloc(&devA, sizeA);
        cudaMalloc(&devB, sizeB);
        cudaMalloc(&devC, sizeC);

        cudaMemcpy(devA, a.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, b.data(), sizeB, cudaMemcpyHostToDevice);

        const dim3 blockSize = getIdealBlockSize(config.m, config.n);
        dim3 gridSize(config.m / blockSize.x + 1, config.n / blockSize.y + 1, 1);

        matrixMultiplication<<<gridSize, blockSize>>>(devA, devB, devC, config.m, config.n, config.l);

        std::vector<FloatType> result(config.m * config.n, 0.0);
        cudaMemcpy(result.data(), devC, sizeC, cudaMemcpyDeviceToHost);

        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
        return result;
    }

    template <typename FloatType>
    dim3 ImplCuda<FloatType>::getIdealBlockSize(const unsigned int x, const unsigned int y) {
        constexpr unsigned int WRAP_SIZE = 32;
        const unsigned int blockSizeLimit = x * y;
        int blockSize = 0;
        int minGridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reinterpret_cast<void *>(matrixMultiplication<float>), 0, blockSizeLimit);
        if (blockSize == blockSizeLimit) {
            return {x, y, 1};
        }
        // blockSize is most likely either 768 (32x24) or 1024 (32x32) given the current GPUs
        // Number of Resident Threads varies between 1024, 1536 and 2048; maximum block size is always 1024
        // Hence, it's either 1x1024 per SM, 2x768 per SM or 2x1024 per SM
        return {WRAP_SIZE, blockSize / WRAP_SIZE, 1};
    }

    /* Explicit Instantiation for float and double */
    template class ImplCuda<float>;
    template class ImplCuda<double>;
} // namespace ppb
