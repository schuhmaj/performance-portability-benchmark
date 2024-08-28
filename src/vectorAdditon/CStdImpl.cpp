#include <algorithm>
#include <benchmark/benchmark.h>
#ifdef __NVCOMPILER
#include <execution>
#endif
#include "VectorAddition.h"


template <typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
#ifdef __NVCOMPILER
    std::transform(std::execution::par_unseq, _inA.begin(), _inA.end(), _inB.begin(), _outC.begin(),
                   std::plus<FloatType>());
#else
    std::transform(_inA.begin(), _inA.end(), _inB.begin(), _outC.begin(), std::plus<FloatType>());
#endif
    checkValidity();
    return _outC;
}

template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)
    ->Name("VecAdd-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
