#include <algorithm>
#include <benchmark/benchmark.h>
#include <openacc.h>
#include "VectorAddition.h"

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>();
}

template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    FloatType *a = _inA.data();
    FloatType *b = _inB.data();
    std::vector<FloatType> result(_size);
    FloatType *c = result.data();
#pragma acc parallel loop copyin(a[0 : _size], b[0 : _size]) copyout(c[0 : _size])
    for (size_t i = 0; i < _size; ++i) {
        c[i] = a[i] + b[i];
    }
    return result;
}

template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)
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

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}