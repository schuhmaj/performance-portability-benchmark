#include "benchmark/benchmark.h"
#include "Impl_Kokkos.h"
#include "Kokkos_Core.hpp"
#include "MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplKokkos<float>>::benchmark)
    ->Name("MatrixMultiplication-Kokkos-Float")
    ->Arg(512)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
