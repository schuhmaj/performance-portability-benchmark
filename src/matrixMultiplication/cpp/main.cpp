#include "benchmark/benchmark.h"
#include "Impl_Cpp.h"
#include "MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCpp<float>>::benchmark)
    ->Name("MatrixMultiplication-CStd-Float")
    ->Arg(4096)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
