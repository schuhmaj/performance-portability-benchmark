#include <benchmark/benchmark.h>
#include "Impl_Kokkos.h"
#include "NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplKokkos<float>>::benchmark)
    ->Name("NBody-Kokkos_SoA_2D_Kernel-Float")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplKokkos<double>>::benchmark)
    ->Name("NBody-Kokkos_SoA_2D_Kernel-Double")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
