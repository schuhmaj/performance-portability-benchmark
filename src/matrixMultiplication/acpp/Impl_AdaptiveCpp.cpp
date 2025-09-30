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

    // We make it shared for easy access to copy results back
    FloatType *deviceResult = sycl::aligned_alloc_shared<FloatType>(ALIGNMENT, resultSize, queue);
    // Simple Kernel
    auto event = queue.submit([&](sycl::handler &h) {
        sycl::range<2> local{16, 16};
        sycl::range<2> global{util::roundUp(M, local[0]), util::roundUp(N, local[1])};
        h.parallel_for(sycl::nd_range<2>{global, local}, [=](const sycl::nd_item<2> &it) {
            const size_t row = it.get_global_id(0);
            const size_t column = it.get_global_id(1);
            if (row >= M || column >= N) return;

            FloatType sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += deviceA[row + k * M] * deviceB[k + column * K];
            }
            deviceResult[row + column * M] = sum;
        });
    });
    // Using Work Groups and Shared Memory
    // auto event = queue.submit([&](sycl::handler &h) {
    //     constexpr size_t TM = 16;
    //     constexpr size_t TN = 16;
    //     constexpr size_t TK = 16;
    //
    //     sycl::range<2> local{TM, TN};
    //     sycl::range<2> global{util::roundUp(M, local[0]), util::roundUp(N, local[1])};
    //
    //     sycl::local_accessor<FloatType, 2> shrA({TM, TK}, h);
    //     sycl::local_accessor<FloatType, 2> shrB({TK, TN}, h);
    //     h.parallel_for(
    //         sycl::nd_range<2>{global, local},
    //         [=](sycl::nd_item<2> it) {
    //             const size_t local_row = it.get_local_id(0);
    //             const size_t local_column = it.get_local_id(1);
    //             const size_t row = it.get_global_id(0);
    //             const size_t column = it.get_global_id(1);
    //
    //             if (row >= M || column >= N) {
    //                 return;
    //             }
    //             FloatType sum = 0.0;
    //             for (size_t k = 0; k < K; k += TK) {
    //                 const size_t kA = k + local_column;
    //                 if (row < M && kA < K) {
    //                     shrA[local_row][local_column] = deviceA[row + kA * M];
    //                 } else {
    //                     shrA[local_row][local_column] = 0.0;
    //                 }
    //
    //                 const size_t kB = k + local_row; // use local row to span TK
    //                 if (kB < K && column < N) {
    //                     shrB[local_row][local_column] = deviceB[kB + column * K];
    //                 } else {
    //                     shrB[local_row][local_column] = 0.0;
    //                 }
    //                 it.barrier(sycl::access::fence_space::local_space);
    //
    //                 for (size_t kk = 0; kk < TK; ++kk) {
    //                     sum += shrA[local_row][kk] * shrB[kk][local_column];
    //                 }
    //                 it.barrier(sycl::access::fence_space::local_space);
    //             }
    //             deviceResult[row + column * M] = sum;
    //         });
    // });
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

