#include <benchmark/benchmark.h>
#include <iostream>
#include <sycl/sycl.hpp>
#include "VectorAddition.h"

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {
    static constexpr size_t ALIGNMENT = 64;

    sycl::queue queue{sycl::default_selector_v};

    FloatType *deviceA;
    FloatType *deviceB;

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b) :
        deviceA{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)},
        deviceB{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)} {
        queue.copy(a.data(), deviceA, size);
        queue.copy(b.data(), deviceB, size);
        queue.wait_and_throw();
    }

    ~impl() {
        sycl::free(deviceA, queue);
        sycl::free(deviceB, queue);
    }
};

template <typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

// Unique kernel name per FloatType to avoid ODR/redefinition
template <typename T>
class VecAddKernel;

template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    const auto &deviceA = _impl->deviceA;
    const auto &deviceB = _impl->deviceB;

    auto *result = sycl::aligned_alloc_shared<FloatType>(_impl->ALIGNMENT, _size, _impl->queue);
    _impl->queue.submit([&](sycl::handler &h) {
        h.parallel_for<VecAddKernel<FloatType>>(sycl::range<1>{_size},
                                                [=](sycl::id<1> i) { result[i] = deviceA[i] + deviceB[i]; });
    });
    _impl->queue.wait_and_throw();

    std::vector<FloatType> hostResult(result, result + _size);
    sycl::free(result, _impl->queue);
    return hostResult;
}

template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)
    ->Name("VecAdd-AdaptiveCpp-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)
    ->Name("VecAdd-AdaptiveCpp-Double")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
