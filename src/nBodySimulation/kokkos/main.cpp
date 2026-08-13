#include <benchmark/benchmark.h>
#include "Impl_Kokkos.h"
#include "Impl_KokkosReduction.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplKokkos<ppb::NBodyBenchmarkConf::float_type>>::benchmark)
    ->Name("NBody-Naive")
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Complexity();

BENCHMARK(ppb::NBodySimulation<ppb::ImplKokkosReduction<ppb::NBodyBenchmarkConf::float_type>>::benchmark)
    ->Name("NBody-Reduction")
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    ppb::NBodyBenchmarkConf::addContext("Kokkos");
    Kokkos::ScopeGuard guard{argc, argv};
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
