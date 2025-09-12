#include <algorithm>
#include <benchmark/benchmark.h>
#include <openacc.h>
#include "VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct ImplOpenAcc{
        using float_type = FloatType;

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            const FloatType *as = a.data();
            const FloatType *bs = b.data();
            std::vector<FloatType> result(size);
            FloatType *c = result.data();
#pragma acc parallel loop copyin(as[0 : size], bs[0 : size]) copyout(c[0 : size])
            for (size_t i = 0; i < size; ++i) {
                c[i] = as[i] + bs[i];
            }
            return result;
        }
    };

    template class ImplOpenAcc<float>;
}


BENCHMARK(ppb::VectorAddition<ppb::ImplOpenAcc<float>>::benchmark)
    ->Name("VecAdd-OpenACC-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
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