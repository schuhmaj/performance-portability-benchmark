#include "matrixMultiplication/cuda/Impl_Cuda.cuh"
#include "matrixMultiplication/cuda/Impl_CudaTensor.cuh"
#include "matrixMultiplication/cuda/Impl_CudaBuffer.cuh"
#include "matrixMultiplication/cuda/Impl_Cublas.cuh"
#include "matrixMultiplication/cuda/Impl_CudaNaive.cuh"
#include "matrixMultiplication/MatrixMultiplication.h"
#include "benchmark/benchmark.h"

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCublas<ppb::MatrixMultiplicationBenchmarkConf::float_type>>::benchmark)
    ->Name("MatrixMultiplication-Cublas")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaNaive<ppb::MatrixMultiplicationBenchmarkConf::float_type>>::benchmark)
    ->Name("MatrixMultiplication-Naive")
    ->RangeMultiplier(2)
    ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCuda<ppb::MatrixMultiplicationBenchmarkConf::float_type>>::benchmark)
//     ->Name("MatrixMultiplication-SharedMemory")
//     ->RangeMultiplier(2)
//     ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

// BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaBuffer<float>>::benchmark)
//     ->Name("MatrixMultiplication-Buffer")
//     ->RangeMultiplier(2)
//     ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

// BENCHMARK(ppb::MatrixMultiplication<ppb::ImplCudaTensor<ppb::MatrixMultiplicationBenchmarkConf::float_type>>::benchmark)
//     ->Name("MatrixMultiplication-Tensor")
//     ->RangeMultiplier(2)
//     ->Range(ppb::MatrixMultiplicationBenchmarkConf::MIN_SIZE, ppb::MatrixMultiplicationBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

int main(int argc, char** argv) {
    ppb::MatrixMultiplicationBenchmarkConf::addContext("Cuda");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
