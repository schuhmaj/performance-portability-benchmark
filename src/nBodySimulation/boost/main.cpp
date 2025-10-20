#include <benchmark/benchmark.h>
#include "Impl_Boost.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImpBoost<float>>::benchmark)
    ->Name("NBody-Float-Boost")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
