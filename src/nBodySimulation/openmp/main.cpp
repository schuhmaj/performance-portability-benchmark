#include <benchmark/benchmark.h>
#include "Impl_OpenMP.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplOpenMP<float>>::benchmark)
    ->Name("NBody-Float-OpenMP")
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
