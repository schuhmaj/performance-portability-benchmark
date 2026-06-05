#include "benchmark/benchmark.h"
#include "Impl_Vulkan.h"
#include "nBodySimulation/NBodySimulation.h"

#ifndef PARADIGM
#define PARADIGM "Vulkan"
#endif


BENCHMARK(ppb::NBodySimulation<ppb::ImplVulkan<float>>::benchmark)
    ->Name("NBody-Naive")
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    ppb::NBodyBenchmarkConf::addContext(PARADIGM);
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
