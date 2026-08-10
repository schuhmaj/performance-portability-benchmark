#include "Impl_AdaptiveCpp.h"
#include "common/UtilityFloatArithmetic.h"
#include <utility>
#include <sycl/sycl.hpp>


template <typename FloatType>
ppb::ImplAdaptiveCpp<FloatType>::ImplAdaptiveCpp()
    : queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}}
{}

template <typename FloatType>
std::pair<std::vector<FloatType>, double> ppb::ImplAdaptiveCpp<FloatType>::operator()(const std::vector<FloatType> &a,
                                                         const std::vector<FloatType> &b,
                                                         const MatrixMultiplicationConfig &config) {
    const size_t M = config.m;
    const size_t N = config.n;
    const size_t K = config.k;
    const size_t resultSize = M * N;
    FloatType *deviceA{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, a.size(), queue)};
    FloatType *deviceB{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, b.size(), queue)};
    queue.copy(a.data(), deviceA, a.size());
    queue.copy(b.data(), deviceB, b.size());

    FloatType *deviceResult = sycl::aligned_alloc_shared<FloatType>(ALIGNMENT, resultSize, queue);
    auto event = queue.submit([&](sycl::handler &h) {
        // SYCL's last range dimension is the fastest-varying one, so the row index is
        // taken from dimension 1. That keeps the accesses to A and C coalesced, matching
        // the CUDA/OpenCL kernels where the row is mapped onto threadIdx.x/get_global_id(0).
        sycl::range<2> local{16, 16};
        sycl::range<2> global{util::roundUp(N, local[0]), util::roundUp(M, local[1])};
        h.parallel_for(sycl::nd_range<2>{global, local}, [=](const sycl::nd_item<2> &it) {
            const size_t column = it.get_global_id(0);
            const size_t row = it.get_global_id(1);
            if (row >= M || column >= N) return;

            FloatType sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += deviceA[row + k * M] * deviceB[k + column * K];
            }
            deviceResult[row + column * M] = sum;
        });
    });
    event.wait_and_throw();
    auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
    double elapsed_nanoseconds = end - start;

    std::vector<FloatType> result(deviceResult, deviceResult + resultSize);
    return std::make_pair(result, elapsed_nanoseconds);
}

/* Explicit Instantiation for float and double */
template class ppb::ImplAdaptiveCpp<float>;
template class ppb::ImplAdaptiveCpp<double>;

