#include "benchmark/benchmark.h"
#include "Impl_Boost.h"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplBoost<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Boost")
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
