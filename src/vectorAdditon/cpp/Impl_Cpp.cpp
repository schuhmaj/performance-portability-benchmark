#include <algorithm>
#include <chrono>
#include <benchmark/benchmark.h>
#include <execution>
#include "vectorAdditon/VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplCpp {
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            std::vector<FloatType> result(a.size());
            const auto start = std::chrono::high_resolution_clock::now();
            std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                           std::plus<FloatType>());
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            return std::make_pair(result, elapsed_nanoseconds);
        }
    };

    template class ImplCpp<float>;
    template class ImplCpp<double>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplCpp<float>>::benchmark)
    ->Name("VecAdd-Float-CPP")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// BENCHMARK(ppb::VectorAddition<ppb::ImplCpp<double>>::benchmark)
//     ->Name("VecAdd-Double-cpp")
//     ->RangeMultiplier(10)
//     ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
