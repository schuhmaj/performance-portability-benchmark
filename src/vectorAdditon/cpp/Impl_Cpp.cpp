#include <algorithm>
#include <benchmark/benchmark.h>
#include <execution>
#include "VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplCpp {
        using float_type = FloatType;

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            std::vector<FloatType> result(a.size());
            std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                           std::plus<FloatType>());
            return result;
        }
    };

    template class ImplCpp<float>;
    template class ImplCpp<double>;
};

BENCHMARK(ppb::VectorAddition<ppb::ImplCpp<float>>::benchmark)
    ->Name("VecAdd-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

BENCHMARK(ppb::VectorAddition<ppb::ImplCpp<double>>::benchmark)
    ->Name("VecAdd-CStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
