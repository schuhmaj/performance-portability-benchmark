#include "benchmark/benchmark.h"
#include "Impl_Vulkan.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplVulkan<float>>::benchmark)
    ->Name("NBody-Float-Vulkan")
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
