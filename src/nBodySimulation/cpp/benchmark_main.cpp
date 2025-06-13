#include "NBodySimulation.h"
// Bad Style, but functional
#include "Impl_Cpp.cpp"


BENCHMARK(ppb::NBodySimulation<float>::benchmark)
    ->Name("NBody-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<double>::benchmark)
    ->Name("NBody-CStd-Double")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e3)
    ->Unit(benchmark::kMillisecond)
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}