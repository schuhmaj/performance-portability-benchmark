#include "benchmark/benchmark.h"
#include "Impl_OpenMPDevice.h"
#include "MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenMPDevice<float>>::benchmark)
    ->Name("MatrixMultiplication-OpenMP_Device-Float")
    ->Arg(2048)
    ->Iterations(3)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
