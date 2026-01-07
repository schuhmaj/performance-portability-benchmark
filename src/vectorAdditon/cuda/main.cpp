#include <benchmark/benchmark.h>
// #include <likwid-marker.h>

#include "vectorAdditon/cuda/Implementations.cuh"
#include "vectorAdditon/VectorAddition.h"
#include "vectorAdditon/cuda/Implementations.cuh"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::VectorAddition<ppb::ImplCuda<float>>::benchmark)
    ->Name("VecAdd-Float-Cuda")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
    #ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
    #endif
    ->Complexity();


BENCHMARK(ppb::VectorAddition<ppb::ImplCublas<float>>::benchmark)
    ->Name("VecAdd-Float-Cublas")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// BENCHMARK(ppb::VectorAddition<ppb::ImplChunkedCuda<float>>::benchmark)
//     ->Name("VecAdd-Float-Cuda-Chunked")
//     ->RangeMultiplier(10)
//     ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();
//
// BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<float>>::benchmark)
//     ->Name("VecAdd-Float-Thrust")
//     ->RangeMultiplier(10)
//     ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

// BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<double>>::benchmark)
//     ->Name("VecAdd-Double-Thrust")
//     ->RangeMultiplier(10)
//     ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

// Exceute with likwid-perfctr -G 0 -W FLOPS_SP -m src/vectorAdditon/cuda/vec_cuda
int main(int argc, char **argv) {
    // NVMON_MARKER_INIT;
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    // NVMON_MARKER_CLOSE;
}
