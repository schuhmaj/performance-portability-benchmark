#include "benchmark/benchmark.h"
#include "matrixMultiplication/alpaka/Impl_Alpaka.h"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplAlpaka<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Alpaka")
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
