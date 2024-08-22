#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include "boost/compute.hpp"


template<typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    namespace compute = boost::compute;
    // Step 1: Create the compute context and the queue
    compute::device gpu = compute::system::default_device();
    compute::context ctx{gpu};
    compute::command_queue queue{ctx, gpu};

    // Step 2: Create memory on the device
    compute::vector<FloatType> deviceA(_inA.size(), ctx);
    compute::vector<FloatType> deviceB(_inB.size(), ctx);
    compute::vector<FloatType> deviceC(_outC.size(), ctx);

    // Step 3: Copy data from host to device
    compute::copy(_inA.begin(), _inA.end(), deviceA.begin(), queue);
    compute::copy(_inB.begin(), _inB.end(), deviceB.begin(), queue);

    // Step 4: Perform operations on device vectors
    // One could also use compute::plus<FloatType>(). However, this is a nice example how to define custom functions
    BOOST_COMPUTE_FUNCTION(FloatType, add_numbers, (FloatType a, FloatType b), { return a + b; });
    compute::transform(deviceA.begin(), deviceA.end(),
                       deviceB.begin(),
                       deviceC.begin(),
                       add_numbers,
                       queue);

    // Step 5: Copy result back to host
    compute::copy(deviceC.begin(), deviceC.end(), _outC.begin(),queue);
    queue.finish();
    return _outC;
}


template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)->Name("Vector Addition")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

// One does not have a dedicated double example here, as Boost.Compute only supports OpenCL types
// Ergo, one only have int and float as potential template specializations

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}