#include <benchmark/benchmark.h>
// #include <likwid-marker.h>

#include "vectorAdditon/cuda/Implementations.cuh"
#include "vectorAdditon/VectorAddition.h"
#include "vectorAdditon/cuda/Implementations.cuh"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::VectorAddition<ppb::ImplCuda<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd-Naive")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
    #ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
    #endif
    ->Complexity();


BENCHMARK(ppb::VectorAddition<ppb::ImplCublas<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd-Cublas")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// BENCHMARK(ppb::VectorAddition<ppb::ImplChunkedCuda<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
//     ->Name("VecAdd-Chunked")
//     ->RangeMultiplier(10)
//     ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();
//
// BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
//     ->Name("VecAdd-Thrust")
//     ->RangeMultiplier(10)
//     ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();


// Exceute with likwid-perfctr -G 0 -W FLOPS_SP -m src/vectorAdditon/cuda/vec_cuda
int main(int argc, char **argv) {
    // NVMON_MARKER_INIT;
    ppb::VectorAdditionBenchmarkConf::addContext("Cuda");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    // NVMON_MARKER_CLOSE;
}
