#include "benchmark/benchmark.h"
#include "Impl_Kokkos.h"
#include "Kokkos_Core.hpp"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplKokkos<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Kokkos")
    ->RangeMultiplier(2)
    ->Range(32, 8192)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
