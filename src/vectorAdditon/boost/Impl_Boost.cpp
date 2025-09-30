#include <chrono>
#include <utility>
#include <benchmark/benchmark.h>
#include "vectorAdditon/VectorAddition.h"
#include "boost/compute.hpp"

namespace ppb {
    template <typename FloatType>
    struct ImplBoost {

        using float_type = FloatType;

        boost::compute::device gpu = boost::compute::system::default_device();
        boost::compute::context context{gpu};
        boost::compute::command_queue queue{context, gpu};

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            boost::compute::vector<FloatType> deviceA{size, context};
            boost::compute::vector<FloatType> deviceB{size, context};
            boost::compute::copy(a.begin(), a.end(), deviceA.begin(), queue);
            boost::compute::copy(b.begin(), b.end(), deviceB.begin(), queue);

            boost::compute::vector<FloatType> resultBuffer(size, context);

            BOOST_COMPUTE_FUNCTION(FloatType, add_numbers, (FloatType a, FloatType b), { return a + b; });
            const auto start = std::chrono::high_resolution_clock::now();
            boost::compute::transform(deviceA.begin(), deviceA.end(),
                               deviceB.begin(),
                               resultBuffer.begin(),
                               add_numbers,
                               queue);
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

            std::vector<FloatType> result(size);
            boost::compute::copy(resultBuffer.begin(), resultBuffer.end(), result.begin(), queue);
            queue.finish();
            return std::make_pair(result, elapsed_nanoseconds);
        }
    };

    template class ImplBoost<float>;
    template class ImplBoost<double>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplBoost<float>>::benchmark)
    ->Name("VecAdd-Float-BoostCL")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

BENCHMARK(ppb::VectorAddition<ppb::ImplBoost<double>>::benchmark)
    ->Name("VecAdd-Double-BoostCL")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char** argv) {
    namespace compute = boost::compute;
    compute::device gpu = compute::system::default_device();
    std::cout << "GPU Name: " << gpu.name() << '\n';

    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}