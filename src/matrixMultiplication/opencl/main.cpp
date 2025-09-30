#include "Impl_OpenCL.h"
#include "benchmark/benchmark.h"
#include "matrixMultiplication/MatrixMultiplication.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplOpenCL<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-ImplOpenCL")
    ->Arg(4096)
    ->Iterations(3)
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
