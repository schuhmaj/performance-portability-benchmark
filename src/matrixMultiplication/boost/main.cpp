#include "benchmark/benchmark.h"
#include "Impl_Boost.h"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplBoost<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Boost")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
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
