#include <benchmark/benchmark.h>
#include <iostream>
#include "RAJA/RAJA.hpp"
#include "VectorAddition.h"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
using exec_policy = RAJA::cuda_exec<256>;
RAJA::resources::Cuda resource;
#else
using exec_policy = RAJA::seq_exec;
RAJA::resources::Host resource;
#endif

template <typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    const size_t size = _inA.size() * sizeof(FloatType);
    // Step 1: Allocate and copy data in RAJA resources (Host to Device)
    FloatType *deviceA = resource.allocate<FloatType>(_inA.size());
    FloatType *deviceB = resource.allocate<FloatType>(_inB.size());
    FloatType *deviceC = resource.allocate<FloatType>(_outC.size());

    resource.memcpy(deviceA, _inA.data(), size);
    resource.memcpy(deviceB, _inB.data(), size);

    // Step 2: Perform vector addition on device
    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, _inA.size()),
                              [=] RAJA_HOST_DEVICE(int i) { deviceC[i] = deviceA[i] + deviceB[i]; });

    // Step 3: Copy the result back to host
    resource.memcpy(_outC.data(), deviceC, size);

    resource.deallocate(deviceA);
    resource.deallocate(deviceB);
    resource.deallocate(deviceC);

    checkValidity();
    return _outC;
}

// Instantiate a benchmark using single precision
template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)
    ->Name("VecAdd-RAJA-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

// Instantiate a benchmark using double precision
template std::vector<double> VectorAddition<double>::operator()();
BENCHMARK(VectorAddition<double>::vectorAdditionBenchmark)
    ->Name("VecAdd-RAJA-Double")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
#if defined(RAJA_ENABLE_CUDA)
    std::cout << "Running with CUDA backend" << std::endl;
#else
    std::cout << "Running with Sequential backend" << std::endl;
#endif

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
