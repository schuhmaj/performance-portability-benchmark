#include <benchmark/benchmark.h>
#include "Impl_Cpp.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplCpp<float>>::benchmark)
    ->Name("NBody-Float-cpp")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
