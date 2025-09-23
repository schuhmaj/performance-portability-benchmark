#include <benchmark/benchmark.h>

#include "vectorAdditon/cuda/Implementations.cuh"
#include "vectorAdditon/VectorAddition.h"
#include "vectorAdditon/cuda/Implementations.cuh"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::VectorAddition<ppb::ImplCuda<float>>::benchmark)
->Name("VecAdd-Cuda-Float")
->RangeMultiplier(10)
->Range(1e6, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
->UseManualTime()
#endif
->Complexity();


BENCHMARK(ppb::VectorAddition<ppb::ImplCublas<float>>::benchmark)
    ->Name("VecAdd-Cublas-Float")
    ->RangeMultiplier(10)
    ->Range(1e6, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::VectorAddition<ppb::ImplChunkedCuda<float>>::benchmark)
    ->Name("VecAdd-Cuda-Float")
    ->RangeMultiplier(10)
    ->Range(1e6, 1e9)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<float>>::benchmark)
    ->Name("VecAdd-Thrust-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<double>>::benchmark)
    ->Name("VecAdd-Thrust-Double")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e7)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
