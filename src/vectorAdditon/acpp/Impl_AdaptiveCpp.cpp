#include <utility>
#include <benchmark/benchmark.h>
#include <iostream>
#include <sycl/sycl.hpp>
#include "vectorAdditon/VectorAddition.h"

namespace ppb {

    // Unique kernel name per FloatType to avoid ODR/redefinition
    template <typename T>
    class VecAddKernel;

    template <typename FloatType>
    struct ImplAcpp {
        using float_type = FloatType;
        static constexpr size_t ALIGNMENT = 64;

        sycl::queue queue{sycl::default_selector_v, {}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            FloatType *deviceA{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)};
            FloatType *deviceB{sycl::aligned_alloc_device<FloatType>(ALIGNMENT, size, queue)};
            queue.copy(a.data(), deviceA, size);
            queue.copy(b.data(), deviceB, size);

            auto *result = sycl::aligned_alloc_shared<FloatType>(ALIGNMENT, size, queue);

            auto event = queue.submit([&](sycl::handler &h) {
                h.parallel_for<VecAddKernel<FloatType>>(sycl::range<1>{size},
                                                        [=](sycl::id<1> i) { result[i] = deviceA[i] + deviceB[i]; });
            });
            event.wait_and_throw();
            // Get time and return elpased time in seconds
            auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            double elapsed_nanoseconds = end - start;

            std::vector<FloatType> hostResult(result, result + size);
            sycl::free(deviceA, queue);
            sycl::free(deviceB, queue);
            sycl::free(result, queue);
            return std::make_pair(hostResult, elapsed_nanoseconds);
        }
    };

    template class ImplAcpp<float>;
    template class ImplAcpp<double>;

}

BENCHMARK(ppb::VectorAddition<ppb::ImplAcpp<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("AdaptiveCpp");
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
