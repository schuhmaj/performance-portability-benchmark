#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include "thrust/transform.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"


template<typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    const size_t size = _inA.size();
    // Step 1: Copy data to gpu device
    thrust::device_vector<FloatType> deviceA(_inA.begin(), _inA.end());
    thrust::device_vector<FloatType> deviceB(_inB.begin(), _inB.end());
    thrust::device_vector<FloatType> deviceC(size);

    // Step 2: Addition of vector
    thrust::transform(deviceA.begin(), deviceA.end(), deviceB.begin(), deviceC.begin(), thrust::plus<float>());

    // Step 3: Copy result from device to host
    thrust::copy(deviceC.begin(), deviceC.end(), _outC.begin());

    checkValidity();
    return _outC;
}

// Explicit instantiation and benchmarking setup
template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)->Name("Vector Addition Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

template std::vector<double> VectorAddition<double>::operator()();
BENCHMARK(VectorAddition<double>::vectorAdditionBenchmark)->Name("Vector Addition Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}