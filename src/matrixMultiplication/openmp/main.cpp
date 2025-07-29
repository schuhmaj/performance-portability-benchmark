#include "Impl_OpenMP.h"
#include "MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenMP<float>>::benchmark)
    ->Name("MatrixMultiplication-OpenMP-Float")
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
