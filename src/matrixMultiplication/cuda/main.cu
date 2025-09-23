#include "matrixMultiplication/cuda/Impl_Cuda.cuh"
#include "matrixMultiplication/cuda/Impl_CudaTensor.cuh"
#include "matrixMultiplication/cuda/Impl_CudaBuffer.cuh"
#include "matrixMultiplication/cuda/Impl_Cublas.cuh"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCublas<float>>::benchmark)
    ->Name("MatrixMultiplication-Cublas-Float")
    ->Arg(4096)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCuda<float>>::benchmark)
    ->Name("MatrixMultiplication-Cuda-Float")
    ->Arg(4096)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaBuffer<float>>::benchmark)
    ->Name("MatrixMultiplication-CudaBuffer-Float")
    ->Arg(4096)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaTensor<float>>::benchmark)
    ->Name("MatrixMultiplication-CudaTensor-Float")
    ->Arg(4096)
    ->Iterations(10)
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
