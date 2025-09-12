#include <algorithm>
#include <benchmark/benchmark.h>
#include "VectorAddition.h"
#include "omp.h"

namespace ppb {
    template <typename FloatType>
    struct ImplOpenMP{
        using float_type = FloatType;

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            const FloatType *as = a.data();
            const FloatType *bs = b.data();
            std::vector<FloatType> result(size);
            FloatType *c = result.data();
#pragma omp target parallel for map(to : as[0 : size], bs[0 : size]) map(from : c[0 : size])
            for (size_t i = 0; i < size; ++i) {
                c[i] = as[i] + bs[i];
            }
            return result;
        }
    };

    template class ImplOpenMP<float>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplOpenMP<float>>::benchmark)
    ->Name("VecAdd-OpenMP-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
