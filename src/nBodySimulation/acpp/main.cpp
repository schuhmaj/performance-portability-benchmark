#include <benchmark/benchmark.h>
#include "Impl_AdaptiveCpp.h"
#include "NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<float>>::benchmark)
    ->Name("NBody-AdaptiveCpp-Float")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<double>>::benchmark)
    ->Name("NBody-AdaptiveCpp-Double")
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
