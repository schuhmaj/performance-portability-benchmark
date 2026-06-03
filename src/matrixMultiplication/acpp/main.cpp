#include "benchmark/benchmark.h"
#include "Impl_AdaptiveCpp.h"
#include "Impl_AdaptiveCppShr.h"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplAdaptiveCpp<ppb::MatrixMultiplicationBenchmarkConf::float_type>>::benchmark)
    ->Name("MatrixMultiplication-Naive")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// BENCHMARK(ppb::MatrixMultiplication<ppb::ImplAdaptiveCppShr<ppb::MatrixMultiplicationBenchmarkConf::float_type>>::benchmark)
//     ->Name("MatrixMultiplication-SharedMemory")
//     ->RangeMultiplier(2)
//     ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

int main(int argc, char** argv) {
    ppb::MatrixMultiplicationBenchmarkConf::addContext("AdaptiveCpp");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
