#include "matrixMultiplication/cuda/Impl_Cuda.cuh"
#include "matrixMultiplication/cuda/Impl_CudaTensor.cuh"
#include "matrixMultiplication/cuda/Impl_CudaBuffer.cuh"
#include "matrixMultiplication/cuda/Impl_Cublas.cuh"
#include "matrixMultiplication/cuda/Impl_CudaNaive.cuh"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCublas<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cublas")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaNaive<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cuda-Naive")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCuda<float>>::benchmark)
    ->Name("MatrixMultiplication-Float-Cuda-SharedMemory")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaBuffer<float>>::benchmark)
//     ->Name("MatrixMultiplication-Float-Cuda-Buffer")
//     ->RangeMultiplier(2)
//     ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

// BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaTensor<float>>::benchmark)
//     ->Name("MatrixMultiplication-Float-Cuda-Tensor")
//     ->RangeMultiplier(2)
//     ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
