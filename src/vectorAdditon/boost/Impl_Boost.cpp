#include <benchmark/benchmark.h>
#include "VectorAddition.h"
#include "boost/compute.hpp"

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {

    boost::compute::device gpu = boost::compute::system::default_device();
    boost::compute::context ctx{gpu};
    boost::compute::command_queue queue{ctx, gpu};


    boost::compute::vector<FloatType> deviceA;
    boost::compute::vector<FloatType> deviceB;

    impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b)
    : deviceA{size, ctx},
      deviceB {size, ctx} {
        boost::compute::copy(a.begin(), a.end(), deviceA.begin(), queue);
        boost::compute::copy(b.begin(), b.end(), deviceB.begin(), queue);
    }
};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}


template<typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    std::vector<FloatType> result(_size);
    boost::compute::vector<FloatType> resultBuffer(_size, _impl->ctx);

    BOOST_COMPUTE_FUNCTION(FloatType, add_numbers, (FloatType a, FloatType b), { return a + b; });
    boost::compute::transform(_impl->deviceA.begin(), _impl->deviceA.end(),
                       _impl->deviceB.begin(),
                       resultBuffer.begin(),
                       add_numbers,
                       _impl->queue);

    boost::compute::copy(resultBuffer.begin(), resultBuffer.end(), result.begin(), _impl->queue);
    _impl->queue.finish();
    return result;
}


template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-BoostCL-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)->Name("VecAdd-BoostCL-Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();


int main(int argc, char** argv) {
    namespace compute = boost::compute;
    compute::device gpu = compute::system::default_device();
    std::cout << "GPU Name: " << gpu.name() << '\n';

    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}