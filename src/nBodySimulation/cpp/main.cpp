#include <benchmark/benchmark.h>
#include "Impl_Cpp.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplCpp<float>>::benchmark)
    ->Name("NBody-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplCpp<double>>::benchmark)
    ->Name("NBody-CStd-Double")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
