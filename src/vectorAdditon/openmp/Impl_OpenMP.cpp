#include <algorithm>
#include <chrono>
#include <utility>
#include <benchmark/benchmark.h>
#include "vectorAdditon/VectorAddition.h"
#include "omp.h"

namespace ppb {
    template <typename FloatType>
    struct ImplOpenMP{
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            const FloatType *as = a.data();
            const FloatType *bs = b.data();
            std::vector<FloatType> result(size);
            FloatType *c = result.data();
            const auto start = std::chrono::high_resolution_clock::now();
#pragma omp target parallel for map(to : as[0 : size], bs[0 : size]) map(from : c[0 : size])
            for (size_t i = 0; i < size; ++i) {
                c[i] = as[i] + bs[i];
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_nanoseconds =
                static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            return std::make_pair(result, elapsed_nanoseconds);
        }
    };

    template class ImplOpenMP<float>;
    template class ImplOpenMP<double>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplOpenMP<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("OpenMP");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
