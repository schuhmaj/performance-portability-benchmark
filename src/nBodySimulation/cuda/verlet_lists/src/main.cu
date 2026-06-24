#include <benchmark/benchmark.h>
#include "Impl_Cuda.cuh"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/CSVFileHandler.h"
#include <string>
#include <iostream>

#ifndef NDEBUG
int main(int argc, char** argv) {
    std::string filename = "/u/home/ge95mis/performance-portability-benchmark/input/particles.csv";
    ppb::CSVFileHandler<float>* reader = new ppb::CSVFileHandler<float>(filename);
    std::vector<ppb::Particle<float>> particles = reader->read();     
    ppb::ParticleSimulationConfig<float>* config = new ppb::ParticleSimulationConfig<float>(particles.size(), 100, 0.01f);
    ppb::ImplCuda<float>* impl = new ppb::ImplCuda<float>(*config);
    impl->simulate(particles);
}
#else
BENCHMARK(ppb::NBodySimulation<ppb::ImplCuda<float>>::benchmark)
    ->Name("NBody-Float-Cuda")
    ->Iterations(1)
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
#endif