#include <benchmark/benchmark.h>
#include "../NBodySimulation.h"
#include "Impl_AdaptiveCpp.h"
#include "nBodySimulation/NBodySimulation.h"

template<size_t Value, typename FloatType, ppb::SorterKinds SorterKind>
static void BM_NBodySimulation(benchmark::State &state) {
    using SimType = ppb::NBodySimulation<ppb::ImplAdaptiveCpp<FloatType, ppb::CellList<SorterKind, ppb::CellListConfig::cutoff, Value>>>;

    for (auto _ : state) {
        SimType::benchmark(state);
    }
}

template<size_t Start, size_t End, typename FloatType, ppb::SorterKinds SorterKind>
struct RunBenchmarks {
    static void Run() {
        BENCHMARK_TEMPLATE(BM_NBodySimulation, Start, FloatType, SorterKind)
            ->Name("Param_Search-CellList-"+ppb::to_string(SorterKind)+(std::is_same_v<FloatType, float>?"-Float":"-Double")+"/"+std::to_string(Start))
            ->RangeMultiplier(10)
            ->Range(1e3, 1e5)
            ->Complexity();
        RunBenchmarks<Start * 2, End, FloatType, SorterKind>::Run();
    }
};

template<size_t End, typename FloatType, ppb::SorterKinds SorterKind>
struct RunBenchmarks<End, End, FloatType, SorterKind> {
    static void Run() {
        BENCHMARK_TEMPLATE(BM_NBodySimulation, End, FloatType, SorterKind)
            ->Name("Param_Search-CellList-"+ppb::to_string(SorterKind)+(std::is_same_v<FloatType, float>?"-Float":"-Double")+"/"+std::to_string(End))
            ->RangeMultiplier(10)
            ->Range(1e3, 1e5)
            ->Complexity();
    }
};

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    RunBenchmarks<1,256,float_t,ppb::SorterKinds::None>::Run();
    RunBenchmarks<1,256,double_t,ppb::SorterKinds::None>::Run();
    RunBenchmarks<1,256,float_t,ppb::SorterKinds::MergeCellID>::Run();
    RunBenchmarks<1,256,double_t,ppb::SorterKinds::MergeCellID>::Run();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
