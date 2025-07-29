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
    std::vector<FloatType> ppb::ImplCuda<FloatType>::operator()(const std::vector<FloatType> &a,
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

        dim3 blockSize(32, 32, 1);
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
    int ImplCuda<FloatType>::getIdealBlockSize(const int problemSize) {
        int blockSize = 0;
        int minGridSize = 0;

        //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)matrixMultiplication, 0, problemSize);
        return blockSize;
    }

    /* Explicit Instantiation for float and double */
    template class ImplCuda<float>;
    template class ImplCuda<double>;
} // namespace ppb
