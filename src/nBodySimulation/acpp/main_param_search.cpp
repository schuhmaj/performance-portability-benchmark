#include <benchmark/benchmark.h>
#include "../NBodySimulation.h"
#include "Impl_AdaptiveCpp.h"
#include "nBodySimulation/NBodySimulation.h"

template<size_t Value, size_t Position, ppb::AlgorithmKinds Algorithm, typename FloatType>
static void BM_NBodySimulation(benchmark::State &state) {
    using SimType =
        std::conditional_t<
            (Algorithm == ppb::AlgorithmKinds::Naive && Position == 0),
            ppb::NBodySimulation<ppb::ImplAdaptiveCpp<FloatType, ppb::Naive<Value, ppb::NaiveDefaults[1]>>>,
        std::conditional_t<
            (Algorithm == ppb::AlgorithmKinds::Naive && Position == 1),
            ppb::NBodySimulation<ppb::ImplAdaptiveCpp<FloatType, ppb::Naive<ppb::NaiveDefaults[0], Value>>>,
        std::conditional_t<
            (Algorithm == ppb::AlgorithmKinds::CellList && Position == 2),
            ppb::NBodySimulation<ppb::ImplAdaptiveCpp<FloatType, ppb::CellList<ppb::CellListDefaults[0], ppb::CellListDefaults[1], Value, ppb::CellListDefaults[3]>>>,
        std::conditional_t<
            (Algorithm == ppb::AlgorithmKinds::CellList && Position == 3),
            ppb::NBodySimulation<ppb::ImplAdaptiveCpp<FloatType, ppb::CellList<ppb::CellListDefaults[0], ppb::CellListDefaults[1], ppb::CellListDefaults[2], Value>>>,
    void
        >>>>;

    if constexpr (!std::is_same_v<SimType, void>) {
        for (auto _ : state) {
            SimType::benchmark(state);
        }
    }
}

template<size_t Start, size_t End, size_t Position, ppb::AlgorithmKinds Algorithm, typename FloatType>
struct RunBenchmarks {
    static void Run() {
        BENCHMARK_TEMPLATE(BM_NBodySimulation, Start, Position, Algorithm, FloatType)
            ->Name("Param_Search-"+to_string(Algorithm)+"-"+std::to_string(Position)+(std::is_same_v<FloatType, float>?"-Float":"-Double")+"/"+std::to_string(Start))
            ->RangeMultiplier(10)
            ->Range(1e2, 1e3)
            ->Complexity();
        RunBenchmarks<Start * 2, End, Position, Algorithm, FloatType>::Run();
    }
};

template<size_t End, size_t Position, ppb::AlgorithmKinds Algorithm, typename FloatType>
struct RunBenchmarks<End, End, Position, Algorithm, FloatType> {
    static void Run() {
        BENCHMARK_TEMPLATE(BM_NBodySimulation, End, Position, Algorithm, FloatType)
            ->Name("Param_Search-"+to_string(Algorithm)+"-"+std::to_string(Position)+(std::is_same_v<FloatType, float>?"-Float":"-Double")+"/"+std::to_string(End))
            ->RangeMultiplier(10)
            ->Range(1e2, 1e3)
            ->Complexity();
    }
};

int main(int argc, char** argv) {
    std::string algorithm_arg = "Naive";
    size_t position_arg = 0;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--algorithm" && argc > i + 1) {
            algorithm_arg = argv[i + 1];
        }
        if (std::string(argv[i]) == "--position" && argc > i + 1) {
            position_arg = std::stoi(argv[i + 1]);
        }
    }

    std::string out_file = "--benchmark_out=param_search_"+algorithm_arg+"_"+std::to_string(position_arg)+"_result.json";
    std::vector<char *> args(argv, argv + argc);
    args.push_back(const_cast<char *>(out_file.c_str()));
    int argsc = args.size();

    benchmark::Initialize(&argsc, args.data());
    if (algorithm_arg == "Naive" && position_arg == 0) {
        RunBenchmarks<1, 1024, 0, ppb::AlgorithmKinds::Naive, float>::Run();
    }
    if (algorithm_arg == "Naive" && position_arg == 1) {
        RunBenchmarks<1, 1024, 1, ppb::AlgorithmKinds::Naive, double>::Run();
    }
    if (algorithm_arg == "CellList" && position_arg == 2) {
        RunBenchmarks<4, 1024, 2, ppb::AlgorithmKinds::CellList, float>::Run();
        RunBenchmarks<4, 1024, 2, ppb::AlgorithmKinds::CellList, double>::Run();
    }
    if (algorithm_arg == "CellList" && position_arg == 3) {
        RunBenchmarks<4, 1024, 3, ppb::AlgorithmKinds::CellList, float>::Run();
        RunBenchmarks<4, 1024, 3, ppb::AlgorithmKinds::CellList, double>::Run();
    }
    if (algorithm_arg == "Verlet") {
        // RunBenchmarks<1, 1024, 0, ppb::AlgorithmKinds::Verlet, float>::Run();
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
