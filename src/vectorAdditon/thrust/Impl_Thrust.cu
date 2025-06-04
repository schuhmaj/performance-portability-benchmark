#include <queue>


#include <benchmark/benchmark.h>
#include "VectorAddition.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/host_vector.h"
#include "thrust/transform.h"

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

    thrust::device_vector<FloatType> deviceA;
    thrust::device_vector<FloatType> deviceB;

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b)
    : deviceA{a.begin(), a.end()},
      deviceB {b.begin(), b.end()} {
    }
};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

template<typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    std::vector<FloatType> result(_size);
    thrust::device_vector<FloatType> resultBuffer(_size);

    thrust::transform(thrust::device, _impl->deviceA.begin(), _impl->deviceA.end(),_impl->deviceB.begin(), resultBuffer.begin(), thrust::plus<FloatType>());
    thrust::copy(resultBuffer.begin(), resultBuffer.end(), result.begin());

    return result;
}

// Explicit instantiation and benchmarking setup
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-Thrust-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)->Name("VecAdd-Thrust-Double")->RangeMultiplier(10)->Range(1e3, 1e7)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}