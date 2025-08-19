#include "Impl_Cuda.cuh"
#include "Impl_CudaTensor.cuh"
#include "Impl_CudaBuffer.cuh"
#include "MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCuda<float>>::benchmark)
    ->Name("MatrixMultiplication-Cuda-Float")
    ->Arg(4096)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaBuffer<float>>::benchmark)
    ->Name("MatrixMultiplication-CudaBuffer-Float")
    ->Arg(4096)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaTensor<float>>::benchmark)
    ->Name("MatrixMultiplication-CudaTensor-Float")
    ->Arg(4096)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
