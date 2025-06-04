#include <CL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <iostream>
#include "VectorAddition.h"

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

    static constexpr size_t ALIGNMENT = 64;

    cl::sycl::default_selector selector;
    cl::sycl::device device{selector};
    cl::sycl::queue queue{selector};

    FloatType* deviceA;
    FloatType* deviceB;

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b)
    : deviceA{cl::sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)},
      deviceB {cl::sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)} {
        queue.copy(a.data(), deviceA, size);
        queue.copy(b.data(), deviceB, size);
        queue.wait_and_throw();
    }

    ~impl() {
        cl::sycl::free(deviceA, queue);
        cl::sycl::free(deviceB, queue);
    }
};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

template<typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    const auto& deviceA = _impl->deviceA;
    const auto& deviceB = _impl->deviceA;
    FloatType* result = cl::sycl::aligned_alloc_shared<FloatType>(_impl->ALIGNMENT, _size, _impl->queue);
    _impl->queue.submit([&](cl::sycl::handler& h) {
        h.parallel_for<class VecAdd>(cl::sycl::range<1>{_size}, [=](cl::sycl::id<1> i) {
            result[i] = deviceA[i] + deviceB[i];
        });
    });
    _impl->queue.wait_and_throw();
    return std::vector<FloatType>(result, result + _size);
}


// Instantiate a benchmark using single precision
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-AdaptiveCpp-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

// Instantiate a benchmark using double precision
template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)->Name("VecAdd-AdaptiveCpp-Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}