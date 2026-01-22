#include <benchmark/benchmark.h>
#include "Impl_AdaptiveCpp.h"
#include "nBodySimulation/NBodySimulation.h"

BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::MergeProjection>>>::benchmark)
    ->Name("NBody-Float-AdaptiveCpp-Naive-Sorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<double, ppb::Naive<ppb::SorterKinds::MergeProjection>>>::benchmark)
    ->Name("NBody-Double-AdaptiveCpp-Naive-Sorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<float, ppb::Naive<ppb::SorterKinds::None>>>::benchmark)
    ->Name("NBody-Float-AdaptiveCpp-Naive-Unsorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<double, ppb::Naive<ppb::SorterKinds::None>>>::benchmark)
    ->Name("NBody-Double-AdaptiveCpp-Naive-Unsorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e4)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::MergeCellID>>>::benchmark)
    ->Name("NBody-Float-AdaptiveCpp-CellList-Sorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<double, ppb::CellList<ppb::SorterKinds::MergeCellID>>>::benchmark)
    ->Name("NBody-Double-AdaptiveCpp-CellList-Sorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<float, ppb::CellList<ppb::SorterKinds::None>>>::benchmark)
    ->Name("NBody-Float-AdaptiveCpp-CellList-Unsorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();
BENCHMARK(ppb::NBodySimulation<ppb::ImplAdaptiveCpp<double, ppb::CellList<ppb::SorterKinds::None>>>::benchmark)
    ->Name("NBody-Double-AdaptiveCpp-CellList-Unsorted")
    ->RangeMultiplier(10)
    ->Range(1e1, 1e6)
    ->Complexity();


int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
