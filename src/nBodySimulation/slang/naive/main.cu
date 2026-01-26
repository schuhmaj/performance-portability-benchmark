#include <benchmark/benchmark.h>
#include "Impl_Slang_Cuda.cuh"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplSlangCuda<float>>::benchmark)
    ->Name("NBody-Float-Cuda")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
