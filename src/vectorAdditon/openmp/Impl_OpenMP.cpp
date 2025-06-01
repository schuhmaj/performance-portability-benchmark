#include <algorithm>
#include <benchmark/benchmark.h>
#include "VectorAddition.h"
#include "omp.h"


template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    const size_t size = _inA.size();
    FloatType *a = _inA.data();
    FloatType *b = _inB.data();
    FloatType *c = _outC.data();
#pragma omp target parallel for map(to : a[0 : size], b[0 : size]) map(from : c[0 : size])
    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
    checkValidity();
    return _outC;
}

template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::vectorAdditionBenchmark)
    ->Name("VecAdd-OpenMP-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    // Deactivate Benchmrk
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
