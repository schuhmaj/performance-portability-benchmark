#include "Impl_OpenMP.h"
#include "Impl_OpenMPDevice.h"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "benchmark/benchmark.h"

// BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenMP<float>>::benchmark)
//     ->Name("MatrixMultiplication-Float-OpenMP-CPU")
//     ->RangeMultiplier(2)
//     ->Range(32, 8192)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenMPDevice<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-OpenMP")
    ->RangeMultiplier(2)
    ->Range(32, 8192)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
