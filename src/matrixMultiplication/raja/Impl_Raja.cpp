#include "Impl_Raja.h"

#include <chrono>
#include "RAJA/RAJA.hpp"

namespace ppb {

    namespace {
        constexpr int BLOCK_SIZE = 16;

#if defined(RAJA_ENABLE_CUDA)
        using Resource = RAJA::resources::Cuda;
        using KernelPolicy = RAJA::KernelPolicy<
            RAJA::statement::CudaKernelFixed<BLOCK_SIZE * BLOCK_SIZE,
                RAJA::statement::Tile<1, RAJA::tile_fixed<BLOCK_SIZE>, RAJA::cuda_block_y_loop,
                    RAJA::statement::Tile<0, RAJA::tile_fixed<BLOCK_SIZE>, RAJA::cuda_block_x_loop,
                        RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
                            RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                                RAJA::statement::Lambda<0>>>>>>>;
#elif defined(RAJA_ENABLE_HIP)
        using Resource = RAJA::resources::Hip;
        using KernelPolicy = RAJA::KernelPolicy<
            RAJA::statement::HipKernelFixed<BLOCK_SIZE * BLOCK_SIZE,
                RAJA::statement::Tile<1, RAJA::tile_fixed<BLOCK_SIZE>, RAJA::hip_block_y_loop,
                    RAJA::statement::Tile<0, RAJA::tile_fixed<BLOCK_SIZE>, RAJA::hip_block_x_loop,
                        RAJA::statement::For<1, RAJA::hip_thread_y_loop,
                            RAJA::statement::For<0, RAJA::hip_thread_x_loop,
                                RAJA::statement::Lambda<0>>>>>>>;
#elif defined(RAJA_ENABLE_SYCL)
        using Resource = RAJA::resources::Sycl;
        // RAJA::sycl_global_<0|1|2><WORK_GROUP_SIZE> are the index mappings understood by
        // statement::For inside a SyclKernel. (RAJA::sycl_global_item_<N> exists as well, but
        // belongs to the RAJA::launch API and has no kernel-statement executor.) The dimensions
        // follow sycl::range<3>, so dimension 2 varies fastest: mapping the contiguous
        // column-major row index (argument 0) onto it keeps the accesses coalesced. Both
        // dimensions carry BLOCK_SIZE, giving the same BLOCK_SIZE^2 work group as CUDA/HIP.
        using KernelPolicy = RAJA::KernelPolicy<
            RAJA::statement::SyclKernel<
                RAJA::statement::For<1, RAJA::sycl_global_1<BLOCK_SIZE>,
                    RAJA::statement::For<0, RAJA::sycl_global_2<BLOCK_SIZE>,
                        RAJA::statement::Lambda<0>>>>>;
#elif defined(RAJA_ENABLE_OPENMP)
        using Resource = RAJA::resources::Host;
        using KernelPolicy = RAJA::KernelPolicy<
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
                RAJA::statement::For<0, RAJA::seq_exec,
                    RAJA::statement::Lambda<0>>>>;
#else
        using Resource = RAJA::resources::Host;
        using KernelPolicy = RAJA::KernelPolicy<
            RAJA::statement::For<1, RAJA::seq_exec,
                RAJA::statement::For<0, RAJA::seq_exec,
                    RAJA::statement::Lambda<0>>>>;
#endif
    }

    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplRaja<FloatType>::operator()(const std::vector<FloatType> &a,
                                                                              const std::vector<FloatType> &b,
                                                                              const MatrixMultiplicationConfig &config) {
        const int m = config.m;
        const int n = config.n;
        const int k = config.k;

        Resource res = Resource::get_default();

        FloatType *devA = res.template allocate<FloatType>(static_cast<size_t>(m) * k);
        FloatType *devB = res.template allocate<FloatType>(static_cast<size_t>(k) * n);
        FloatType *devC = res.template allocate<FloatType>(static_cast<size_t>(m) * n);

        res.memcpy(devA, a.data(), static_cast<size_t>(m) * k * sizeof(FloatType));
        res.memcpy(devB, b.data(), static_cast<size_t>(k) * n * sizeof(FloatType));
        res.wait();

        const auto start = std::chrono::steady_clock::now();
        // The matrices are stored in column-major format (like the Kokkos LayoutLeft implementation)
        RAJA::kernel_resource<KernelPolicy>(
                RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, m), RAJA::TypedRangeSegment<int>(0, n)),
                res,
                [=] RAJA_HOST_DEVICE(const int i, const int j) {
                    FloatType sum = 0;
                    for (int entry = 0; entry < k; ++entry) {
                        sum += devA[i + static_cast<size_t>(entry) * m] * devB[entry + static_cast<size_t>(j) * k];
                    }
                    devC[i + static_cast<size_t>(j) * m] = sum;
                });
        res.wait();
        const auto end = std::chrono::steady_clock::now();
        const double nanoseconds = std::chrono::duration<double, std::nano>(end - start).count();

        std::vector<FloatType> result(static_cast<size_t>(m) * n);
        res.memcpy(result.data(), devC, static_cast<size_t>(m) * n * sizeof(FloatType));
        res.wait();

        res.deallocate(devA);
        res.deallocate(devB);
        res.deallocate(devC);
        return std::make_pair(std::move(result), nanoseconds);
    }

    /* Explicit Instantiation for float and double */
    template class ImplRaja<float>;
    template class ImplRaja<double>;
}
