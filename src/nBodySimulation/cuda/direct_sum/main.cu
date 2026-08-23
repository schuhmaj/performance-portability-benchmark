#include <benchmark/benchmark.h>
#include "Impl_Cuda.cuh"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::cuda::nbody::ImplCuda<float>>::benchmark)
    ->Name("NBody-Float-Cuda")
    ->Iterations(1)
    ->RangeMultiplier(5)
    ->Range(ppb::NBodyBenchmarkConf::PPB_NBODY_CUDA_MIN_SIZE, ppb::NBodyBenchmarkConf::PPB_NBODY_CUDA_MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    ppb::NBodyBenchmarkConf::addContext("Cuda");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
