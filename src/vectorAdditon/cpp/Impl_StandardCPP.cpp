#include <algorithm>
#include <benchmark/benchmark.h>
#include <execution>
#include "VectorAddition.h"


template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    std::transform(_inA.begin(), _inA.end(), _inB.begin(), _outC.begin(),
                   std::plus<FloatType>());
    checkValidity();
    return _outC;
}

template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::vectorAdditionBenchmark)
    ->Name("VecAdd-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
