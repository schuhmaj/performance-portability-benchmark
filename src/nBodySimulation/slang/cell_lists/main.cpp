#include "benchmark/benchmark.h"
#include "Impl_Slang_Vulkan.h"
#include "nBodySimulation/NBodySimulation.h"

#ifndef GIT_HASH_STRING
#define GIT_HASH_STRING "unknown"
#endif

BENCHMARK(ppb::NBodySimulation<ppb::ImplSlangVulkan<float>>::benchmark)
    ->Name("NBody-Float-Vulkan-Cell-Lists")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e7)
    ->Repetitions(3)
    ->ReportAggregatesOnly(false)
    ->Complexity();

int main(int argc, char** argv) {

    benchmark::AddCustomContext("git_hash", GIT_HASH_STRING);

    const uint32_t wg = ppb::ParticleSimulationConfig<float>::TILE_SIZE;
    benchmark::AddCustomContext("workgroup_size", std::to_string(wg));

    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
