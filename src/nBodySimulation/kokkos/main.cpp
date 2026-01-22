#include <benchmark/benchmark.h>
#include "Impl_Kokkos.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplKokkos<float>>::benchmark)
    ->Name("NBody-Float-Kokkos-SoA_2D_Kernel_TeamPolicy")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Complexity();

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
