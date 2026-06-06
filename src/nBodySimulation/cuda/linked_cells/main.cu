#include <benchmark/benchmark.h>
#include "Impl_Cuda.cuh"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/CSVFileHandler.h"
#include <string>
#include <iostream>

#ifndef PPB_ENABLE_VTK
BENCHMARK(ppb::NBodySimulation<ppb::ImplCuda<float>>::benchmark)
    ->Name("NBody-Float-Cuda")
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
#else
int main(int argc, char** argv) {
    std::string filename = "/mnt/e/Code/Uni/BA/performance-portability-benchmark/input/particles.csv";
    ppb::CSVFileHandler<float>* reader = new ppb::CSVFileHandler<float>(filename);
    std::vector<ppb::Particle<float>> particles = reader->read(); 
    for (auto p : particles) {
        std::cout<<p<<std::endl;
    }
    
    ppb::ParticleSimulationConfig<float>* config = new ppb::ParticleSimulationConfig<float>(particles.size(), 1000, 0.1f);
    ppb::ImplCuda<float>* impl = new ppb::ImplCuda<float>(*config);
    impl->simulate(particles);
}
#endif
