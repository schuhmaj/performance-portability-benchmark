#include <benchmark/benchmark.h>
#include "Impl_AdaptiveCpp.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<float, ppb::Naive<>>>::benchmark)
    ->Name("NBody-Float-AdaptiveCpp-Naive")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<double, ppb::Naive<>>>::benchmark)
    ->Name("NBody-Double-AdaptiveCpp-Naive")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<float, ppb::CellList<>>>::benchmark)
    ->Name("NBody-Float-AdaptiveCpp-CellList")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<double, ppb::CellList<>>>::benchmark)
    ->Name("NBody-Double-AdaptiveCpp-CellList")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
