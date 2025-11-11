#include <benchmark/benchmark.h>
#include <likwid-marker.h>

#include "vectorAdditon/cuda/Implementations.cuh"
#include "vectorAdditon/VectorAddition.h"
#include "vectorAdditon/cuda/Implementations.cuh"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::VectorAddition<ppb::ImplCuda<float>>::benchmark)
->Name("VecAdd-Float-Cuda")
->Args(1e6)
#ifdef PPB_MEASURE_ONLY_KERNEL
->UseManualTime()
#endif
->Complexity();

int main(int argc, char **argv) {
    NVMON_MARKER_INIT;
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    NVMON_MARKER_CLOSE;
}
