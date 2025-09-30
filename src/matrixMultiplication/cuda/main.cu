#include "matrixMultiplication/cuda/Impl_Cuda.cuh"
#include "matrixMultiplication/cuda/Impl_CudaTensor.cuh"
#include "matrixMultiplication/cuda/Impl_CudaBuffer.cuh"
#include "matrixMultiplication/cuda/Impl_Cublas.cuh"
#include "matrixMultiplication/cuda/Impl_CudaNaive.cuh"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCublas<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cublas")
->Arg(4096)->Arg(512)
    ->Iterations(10)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaNaive<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cuda-Naive")
    ->Arg(4096)->Arg(512)
    ->Iterations(10)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCuda<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cuda-SharedMemory")
->Arg(4096)->Arg(512)
    ->Iterations(10)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaBuffer<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cuda-Buffer")
->Arg(4096)->Arg(512)
    ->Iterations(10)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaTensor<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cuda-Tensor")
    ->Arg(4096)
    ->Iterations(10)
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
