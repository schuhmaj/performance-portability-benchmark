#include <benchmark/benchmark.h>
#include "Impl_Cuda.cuh"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/CSVFileHandler.h"
#include <string>
#include <iostream>

BENCHMARK(ppb::NBodySimulation<ppb::cuda::nbody::ImplCuda<float>>::benchmark)
    ->Name("NBody-Float-Cuda")
    ->Iterations(1)
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::PPB_NBODY_CUDA_MIN_SIZE, ppb::NBodyBenchmarkConf::PPB_NBODY_CUDA_MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}