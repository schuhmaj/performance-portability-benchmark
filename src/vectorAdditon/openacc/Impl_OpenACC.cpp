#include <algorithm>
#include <chrono>
#include <utility>
#include "benchmark/benchmark.h"
#include <openacc.h>
#include "vectorAdditon/VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplOpenAcc{
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            const FloatType *as = a.data();
            const FloatType *bs = b.data();
            std::vector<FloatType> result(size);
            FloatType *c = result.data();
            const auto start = std::chrono::high_resolution_clock::now();
#pragma acc parallel loop copyin(as[0 : size], bs[0 : size]) copyout(c[0 : size])
            for (size_t i = 0; i < size; ++i) {
                c[i] = as[i] + bs[i];
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_nanoseconds =
                static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            return std::make_pair(result, elapsed_nanoseconds);
        }
    };

    template class ImplOpenAcc<float>;
}


BENCHMARK(ppb::VectorAddition<ppb::ImplOpenAcc<float>>::benchmark)
    ->Name("VecAdd-Float-OpenACC")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    // Get number of devices before any parallel regions
    int num_devices = acc_get_num_devices(acc_device_default);
    printf("Number of available devices %d\n", num_devices);

    // Fetch the device number outside region
    int device_num = acc_get_device_num(acc_device_default);
    printf("Running on device number %d\n", device_num);

    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}