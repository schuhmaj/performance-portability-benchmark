#include "benchmark/benchmark.h"
#include "Impl_AdaptiveCpp.h"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplAdaptiveCpp<float>>::benchmark)
    ->Name("MatrixMultiplication-AdaptiveCpp-Float")
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
