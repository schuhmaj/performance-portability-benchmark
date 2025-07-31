#include "Impl_Cuda.cuh"

namespace ppb {

    // __global__ void kernel_matrixMultOverlapped(const float *__restrict__ devA,
    //                                         const float *__restrict__ devB,
    //                                         float *__restrict__ devC, const size_t size) {
    //     const int TILE_SIZE = 64;
    //     __shared__ float shrA[TILE_SIZE][TILE_SIZE];
    //     __shared__ float shrB[TILE_SIZE][TILE_SIZE];
    //
    //     const int tx = threadIdx.x;
    //     const int ty = threadIdx.y;
    //
    //     const int row = blockIdx.x * TILE_SIZE + tx;
    //     const int column = blockIdx.y * TILE_SIZE + ty;
    //     if ((row < size) && (column < size)) {
    //
    //         float Celem = 0.0f, Aelem = 0.0f, Belem = 0.0f;
    //         // load the first tile into registers
    //         Aelem = devA[row + size * ty];
    //         Belem = devB[tx + column * size];
    //
    //         for (int m = 0; m < (size / TILE_SIZE) - 1; ++m) {
    //             // load tiles of A and B to the shared mem.
    //             shrA[ty][tx] = Aelem;
    //             shrB[ty][tx] = Belem;
    //             __syncthreads();
    //
    //             // load the next tile to the registers
    //             Aelem = devA[row + size * (ty + (m + 1) * TILE_SIZE)];
    //             Belem = devB[(tx + (m + 1) * TILE_SIZE) + column * size];
    //
    //             for (int j = 0; j < TILE_SIZE; ++j)
    //                 Celem += shrA[j][tx] * shrB[ty][j];
    //             __syncthreads();
    //         };
    //
    //         // compute the last tile
    //         const int m = (size / TILE_SIZE) - 1;
    //         shrA[ty][tx] = Aelem;
    //         shrB[ty][tx] = Belem;
    //         __syncthreads();
    //         for (int j = 0; j < TILE_SIZE; ++j)
    //             Celem += shrA[j][tx] * shrB[ty][j];
    //
    //         devC[row + size * column] += Celem;
    //     }
    // }
    //

    template<typename FloatType>
    __global__ void matrixMultiplication(const FloatType *a, const FloatType *b, FloatType *c, const int m, const int n,
                                         const int k) {
        extern __shared__ float shrB[];
        const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int column = blockIdx.y * blockDim.y + threadIdx.y;
        if (row >= n || column >= m) {
            return;
        }
        FloatType sum = 0.0;
        for (unsigned int entry = 0; entry < k; ++entry) {
            sum += a[row + entry * m] * b[entry + column * k];
        }
        c[row + column * m] =sum;
    }

    template __global__ void matrixMultiplication<float>(const float*, const float*, float*, int, int, int);


    size_t get1DGrid(size_t blockSize, size_t matrixSize) {
        return (matrixSize + blockSize - 1) / blockSize;
    }

    template <typename FloatType>
    std::vector<FloatType> ImplCuda<FloatType>::operator()(const std::vector<FloatType> &a,
                                                                const std::vector<FloatType> &b,
                                                                const MatrixMultiplicationConfig &config) {

        cudaStream_t stream;
        cudaStreamCreate(&stream);

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
        cudaMemsetAsync(&devC, 0, sizeC, stream);

        cudaMemcpyAsync(devA, a.data(), sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devB, b.data(), sizeB, cudaMemcpyHostToDevice, stream);


        // constexpr int tileSize = 64;
        // const size_t shrMemSize = 2 * tileSize * tileSize * sizeof(float);
        // dim3 dimBlock(tileSize, tileSize);
        // const size_t Grid1D = get1DGrid(dimBlock.x, config.n);
        // dim3 dimGrid(Grid1D, Grid1D);
        // kernel_matrixMultCoalescedDym<<<dimGrid, dimBlock, shrMemSize, stream>>>(devA, devB, devC, config.n);
        //kernel_matrixMultOverlapped<<<dimGrid, dimBlock, shrMemSize, stream>>>(devA, devB, devC, config.n);


        matrixMultiplication<<<gridSize, blockSize, config.k * sizeof(FloatType), stream>>>(devA, devB, devC, config.m, config.n, config.k);

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
    //template class ImplCuda<double>;
} // namespace ppb
