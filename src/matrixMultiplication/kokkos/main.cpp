#include "benchmark/benchmark.h"
#include "Impl_Kokkos.h"
#include "Kokkos_Core.hpp"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplKokkos<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Kokkos")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
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
