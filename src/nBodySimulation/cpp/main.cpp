#include <benchmark/benchmark.h>
#include "Impl_Cpp.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplCpp<float>>::benchmark)
    ->Name("NBody-Float-CPP")
    ->RangeMultiplier(10)
    ->Range(ppb::NBodyBenchmarkConf::MIN_SIZE, ppb::NBodyBenchmarkConf::MAX_SIZE)
    ->Repetitions(3)
    ->ReportAggregatesOnly(false)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::AddCustomContext("git_hash", GIT_HASH_STRING);

    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
