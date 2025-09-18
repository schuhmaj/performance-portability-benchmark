#include "Impl_OpenMP.h"
#include "Impl_OpenMPDevice.h"
#include "MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenMP<float>>::benchmark)
    ->Name("MatrixMultiplication-OpenMP-Float")
    ->Arg(4096)
    ->Iterations(3)
    ->Unit(benchmark::kMillisecond)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenMPDevice<float>>::benchmark)
    ->Name("MatrixMultiplication-OpenMP_Device-Float")
    ->Arg(4096)
    ->Iterations(3)
    ->Unit(benchmark::kMillisecond)
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
