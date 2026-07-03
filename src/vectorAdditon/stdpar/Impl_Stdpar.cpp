#include <algorithm>
#include <chrono>
#include <execution>
#include <utility>
#include <benchmark/benchmark.h>

#include "vectorAdditon/VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplStdpar {
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            std::vector<FloatType> result(a.size());
            const auto start = std::chrono::high_resolution_clock::now();
            std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(), result.begin(),
                           std::plus<FloatType>());
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            return std::make_pair(result, elapsed_nanoseconds);
        }
    };

    template class ImplStdpar<float>;
    template class ImplStdpar<double>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplStdpar<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("Stdpar");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
