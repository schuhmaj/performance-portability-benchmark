#include <algorithm>
#include <benchmark/benchmark.h>
#include <execution>
#include "VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplNvHpcStd {
        using float_type = FloatType;
        std::vector<FloatType> :operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            std::vector<FloatType> result(size);
            std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(), result.begin(),
                           std::plus<FloatType>());
            return result;
        }

    };

    template class ImplNvHpcStd<float>;
}


BENCHMARK(ppb::VectorAddition<ImplNvHpcStd<float>>::benchmark)
    ->Name("VecAdd-NvhpcCStd-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
