#include <CL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <iostream>
#include "VectorAddition.h"

template<typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    cl::sycl::default_selector selector;
    cl::sycl::device device{selector};
    cl::sycl::queue q{selector};

    const int size = _inA.size();

    {
        cl::sycl::buffer<FloatType, 1> A(_inA.data(), size);
        cl::sycl::buffer<FloatType, 1> B(_inB.data(), size);
        cl::sycl::buffer<FloatType, 1> C(_outC.data(), size);

        q.submit([&](cl::sycl::handler& h) {
            auto deviceA = A.template get_access<cl::sycl::access::mode::read>(h);
            auto deviceB = B.template get_access<cl::sycl::access::mode::read>(h);
            auto deviceC = C.template get_access<cl::sycl::access::mode::write>(h);

            h.parallel_for<class vectorAddition>(cl::sycl::range<1>(size), [=](const cl::sycl::item<1> &i) {
                deviceC[i] = deviceA[i] + deviceB[i];
            });
        });

        q.wait_and_throw();
    }
    checkValidity();
    return _outC;
}


// Instantiate a benchmark using single precision
template std::vector<float> VectorAddition<float>::operator()();
BENCHMARK(VectorAddition<float>::vectorAdditionBenchmark)->Name("VecAdd-AdaptiveCpp-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}