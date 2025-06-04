#include <algorithm>
#include <benchmark/benchmark.h>
#include "VectorAddition.h"
#include "omp.h"

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>();
}

template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    FloatType *a = _inA.data();
    FloatType *b = _inB.data();
    std::vector<FloatType> result(_size);
    FloatType *c = result.data();
#pragma omp target parallel for map(to : a[0 : _size], b[0 : _size]) map(from : c[0 : _size])
    for (size_t i = 0; i < _size; ++i) {
        c[i] = a[i] + b[i];
    }
    return result;
}

template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)
    ->Name("VecAdd-OpenMP-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
