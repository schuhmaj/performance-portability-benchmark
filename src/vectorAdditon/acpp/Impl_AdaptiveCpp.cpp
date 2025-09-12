#include <benchmark/benchmark.h>
#include <iostream>
#include <sycl/sycl.hpp>
#include "VectorAddition.h"

namespace ppb {

    // Unique kernel name per FloatType to avoid ODR/redefinition
    template <typename T>
    class VecAddKernel;

    template <typename FloatType>
    struct ImplAcpp {
        using float_type = FloatType;
        static constexpr size_t ALIGNMENT = 64;

        sycl::queue queue{sycl::default_selector_v, {}, sycl::property::queue::in_order{}};

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            FloatType *deviceA{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)};
            FloatType *deviceB{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)};
            queue.copy(a.data(), deviceA, size);
            queue.copy(b.data(), deviceB, size);

            auto *result = sycl::aligned_alloc_shared<FloatType>(ALIGNMENT, size, queue);
            queue.submit([&](sycl::handler &h) {
                h.parallel_for<VecAddKernel<FloatType>>(sycl::range<1>{size},
                                                        [=](sycl::id<1> i) { result[i] = deviceA[i] + deviceB[i]; });
            });
            queue.wait_and_throw();

            std::vector<FloatType> hostResult(result, result + size);
            sycl::free(deviceA, queue);
            sycl::free(deviceB, queue);
            sycl::free(result, queue);
            return hostResult;
        }
    };

    template class ImplAcpp<float>;
    template class ImplAcpp<double>;

}

BENCHMARK(ppb::VectorAddition<ppb::ImplAcpp<float>>::benchmark)
    ->Name("VecAdd-AdaptiveCpp-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

BENCHMARK(ppb::VectorAddition<ppb::ImplAcpp<double>>::benchmark)
    ->Name("VecAdd-AdaptiveCpp-Double")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
