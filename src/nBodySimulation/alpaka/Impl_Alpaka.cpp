
#include "Impl_Alpaka.h"
#include <chrono>

template<typename FloatType>
ppb::ImplAlpaka<FloatType>::ImplAlpaka()
    : host(alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0))
    , device(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0))
    , queue(device) {}

template <typename FloatType>
std::pair<std::vector<FloatType>, double>
ppb::ImplAlpaka<FloatType>::operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b,
                                 const MatrixMultiplicationConfig &config) {
    const size_t resultSize = config.m * config.n;
    std::vector<FloatType> result(resultSize, 0.0);

    using Dim1D = alpaka::DimInt<1u>;
    const alpaka::Vec<Dim1D, Idx> extentA(config.m * config.k);
    const alpaka::Vec<Dim1D, Idx> extentB(config.k * config.n);
    const alpaka::Vec<Dim1D, Idx> extentC(config.m * config.n);

    const auto bufHostA = alpaka::createView(host, const_cast<FloatType*>(a.data()), extentA);
    const auto bufHostB = alpaka::createView(host, const_cast<FloatType*>(b.data()), extentB);
    auto bufDevA = alpaka::allocBuf<FloatType, Idx>(device, extentA);
    auto bufDevB = alpaka::allocBuf<FloatType, Idx>(device, extentB);
    auto bufDevC = alpaka::allocBuf<FloatType, Idx>(device, extentC);

    alpaka::memcpy(queue, bufDevA, bufHostA, extentA);
    alpaka::memcpy(queue, bufDevB, bufHostB, extentB);

    // Create Matrix Multiplication Kernel (column-major)
    const auto matMulKernel = [=] ALPAKA_FN_ACC (const Acc& acc,
                                                  const FloatType* const A,
                                                  const FloatType* const B,
                                                  FloatType* const C,
                                                  const Idx M,
                                                  const Idx N,
                                                  const Idx K) {
        const auto globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        const Idx row = globalThreadIdx[0];
        const Idx col = globalThreadIdx[1];

        if (row < M && col < N) {
            FloatType acc_val = 0.0;
            for (Idx k = 0; k < K; ++k) {
                acc_val += A[row + k * M] * B[k + col * K];
            }
            C[row + col * M] = acc_val;
        }
    };
    const alpaka::Vec<Dim, Idx> problemExtent(config.m, config.n);
    const auto workDiv = alpaka::getValidWorkDiv(
        alpaka::KernelCfg<Acc>{problemExtent, alpaka::Vec<Dim, Idx>::ones()},
        device,
        matMulKernel,
        alpaka::getPtrNative(bufDevA),
        alpaka::getPtrNative(bufDevB),
        alpaka::getPtrNative(bufDevC),
        config.m,
        config.n,
        config.k
    );

    auto const taskKernel = alpaka::createTaskKernel<Acc>(
        workDiv,
        matMulKernel,
        alpaka::getPtrNative(bufDevA),
        alpaka::getPtrNative(bufDevB),
        alpaka::getPtrNative(bufDevC),
        config.m,
        config.n,
        config.k
    );

    const auto start = std::chrono::high_resolution_clock::now();
    alpaka::enqueue(queue, taskKernel);
    alpaka::wait(queue);
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());


    auto resultView = alpaka::createView(host, result.data(), extentC);
    alpaka::memcpy(queue, resultView, bufDevC, extentC);
    alpaka::wait(queue);

    return std::make_pair(std::move(result), elapsed_nanoseconds);
}

template class ppb::ImplAlpaka<float>;