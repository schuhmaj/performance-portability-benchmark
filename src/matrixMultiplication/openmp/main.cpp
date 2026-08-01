#include "Impl_OpenMP.h"
#include "Impl_OpenMPDevice.h"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenMPDevice<ppb::MatrixMultiplicationBenchmarkConf::float_type>>::benchmark)
    ->Name("MatrixMultiplication")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char** argv) {
    ppb::MatrixMultiplicationBenchmarkConf::addContext("OpenMP");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
