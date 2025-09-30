#include "Impl_Cuda.cuh"
#include <mma.h>

namespace ppb {

    __global__ void matrixMultiplication(const float *__restrict__ a,
                                           const float *__restrict__ b,
                                           float *__restrict__ c,
                                           const int M, const int N, const int K) {
        // Threadblock shape:
        //  - blockDim.x: number of thread "row groups" (each thread handles 4 rows)
        //  - blockDim.y: number of columns per block
        //  - TILE_K is chosen as blockDim.x for simplicity
        constexpr unsigned int ROWS_PER_THREAD = 4;
        const unsigned int TILE_K = blockDim.x;
        const unsigned int TILE_M = blockDim.x * ROWS_PER_THREAD;
        const unsigned int TILE_N = blockDim.y;

        extern __shared__ float smem[];
        float* __restrict__ tileA = smem;
        float* __restrict__ tileB = tileA + (TILE_M * TILE_K);

        const unsigned int tx = threadIdx.x;
        const unsigned int ty = threadIdx.y;

        const unsigned int baseRow = blockIdx.x * TILE_M + tx * ROWS_PER_THREAD;
        const unsigned int column  = blockIdx.y * blockDim.y + ty;

        const unsigned int numTiles = util::ceilDiv<unsigned int>(K, TILE_K);
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (unsigned int tile = 0; tile < numTiles; ++tile) {
            for (unsigned int kk = ty; kk < TILE_K; kk += blockDim.y) {
                const unsigned int kGlobal = tile * TILE_K + kk;
                const unsigned int shBase = kk * TILE_M + tx * ROWS_PER_THREAD;

                if (kGlobal < K) {
                    const float* ptrA = a + kGlobal * M + baseRow;
                    tileA[shBase + 0] = (baseRow + 0 < M) ? ptrA[0] : 0.0f;
                    tileA[shBase + 1] = (baseRow + 1 < M) ? ptrA[1] : 0.0f;
                    tileA[shBase + 2] = (baseRow + 2 < M) ? ptrA[2] : 0.0f;
                    tileA[shBase + 3] = (baseRow + 3 < M) ? ptrA[3] : 0.0f;
                } else {
                    tileA[shBase + 0] = 0.0f;
                    tileA[shBase + 1] = 0.0f;
                    tileA[shBase + 2] = 0.0f;
                    tileA[shBase + 3] = 0.0f;
                }
            }
            for (unsigned int kk = tx; kk < TILE_K; kk += blockDim.x) {
                const unsigned int kGlobal = tile * TILE_K + kk;
                const unsigned int shIdx = kk * TILE_N + ty;
                tileB[shIdx] = column < N && kGlobal < K ? b[column * K + kGlobal] : 0.0f;;
            }
            __syncthreads();

            // Compute accumulations for this tile
            const int rowOffset = tx * ROWS_PER_THREAD;
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const float bval = tileB[kk * TILE_N + ty];
                const int aBase = kk * TILE_M + rowOffset;
                acc.x += tileA[aBase + 0] * bval;
                acc.y += tileA[aBase + 1] * bval;
                acc.z += tileA[aBase + 2] * bval;
                acc.w += tileA[aBase + 3] * bval;
            }
            __syncthreads();
        }
        if (column < N) {
            if (baseRow + 0 < M) c[baseRow + 0 + column * M] = acc.x;
            if (baseRow + 1 < M) c[baseRow + 1 + column * M] = acc.y;
            if (baseRow + 2 < M) c[baseRow + 2 + column * M] = acc.z;
            if (baseRow + 3 < M) c[baseRow + 3 + column * M] = acc.w;
        }
    }


    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplCuda<FloatType>::operator()(const std::vector<FloatType> &a,
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
        // Shared memory: TILE_K*(TILE_M + TILE_N) floats
        // TILE_K = blockDim.x, TILE_M = blockDim.x*4, TILE_N = blockDim.y
        const size_t sharedMemSize = blockSize.x * (4 * blockSize.x + blockSize.y) * sizeof(FloatType);

        cudaMallocAsync(&devA, sizeA, stream);
        cudaMallocAsync(&devB, sizeB, stream);
        cudaMallocAsync(&devC, sizeC, stream);
        cudaMemsetAsync(&devC, 0, sizeC, stream);

        cudaMemcpyAsync(devA, a.data(), sizeA, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(devB, b.data(), sizeB, cudaMemcpyHostToDevice, stream);

        static_assert(std::is_same_v<FloatType, float>, "This kernel currently supports float only.");
        cudaEventRecord(start, stream);
        matrixMultiplication<<<gridSize, blockSize, sharedMemSize, stream>>>(devA, devB, devC, config.m, config.n, config.k);
        cudaEventRecord(stop, stream);

        std::vector<FloatType> result(config.m * config.n, 0.0);
        cudaMemcpyAsync(result.data(), devC, sizeC, cudaMemcpyDeviceToHost, stream);

        cudaFreeAsync(devA, stream);
        cudaFreeAsync(devB, stream);
        cudaFreeAsync(devC, stream);
        cudaStreamSynchronize(stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        return std::make_pair(result, elapsedTime * 1e6);
    }

    template <typename FloatType>
    dim3 ImplCuda<FloatType>::getIdealBlockSize(const unsigned int m, const unsigned int n) {
        constexpr unsigned int WRAP_SIZE = 32;
        constexpr unsigned int MAX_THREADS = 1024;
        const int blockSizeLimit = static_cast<int>(m) * static_cast<int>(n);
        if (blockSizeLimit <= MAX_THREADS) {
            // Note: With 4 rows/thread, blockDim.x threads cover 4*blockDim.x rows
            // For small problems, keep a compact block; map columns onto y
            unsigned int bx = std::max<unsigned int>(std::min(m / 4 + (m % 4 != 0), WRAP_SIZE), 1);
            unsigned int by = std::max<unsigned int>(std::min<unsigned int>(n, MAX_THREADS / bx), 1);
            return {bx, by, 1};
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
        const int rowsPerBlock = static_cast<int>(blockSize.x) * 4;
        return {util::ceilDiv<unsigned int>(m, rowsPerBlock), util::ceilDiv<unsigned int>(n, blockSize.y), 1};
    }


    /* Explicit Instantiation for float and double */
    template class ImplCuda<float>;
    //template class ImplCuda<double>;
} // namespace ppb
