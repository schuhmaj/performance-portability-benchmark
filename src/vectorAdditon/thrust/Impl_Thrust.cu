#include <benchmark/benchmark.h>
#include <chrono>
#include <utility>
#include <vector>
#include "VectorAddition.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/host_vector.h"
#include "thrust/transform.h"
namespace ppb {
    template <typename FloatType>
    struct ImplThrust {
        using float_type = FloatType;

        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            thrust::device_vector<FloatType> deviceA(a.begin(), a.end());
            thrust::device_vector<FloatType> deviceB(b.begin(), b.end());
            std::vector<FloatType> result(size);
            thrust::device_vector<FloatType> resultBuffer(size);

            const auto start = std::chrono::high_resolution_clock::now();
            thrust::transform(thrust::device, deviceA.begin(), deviceA.end(),deviceB.begin(), resultBuffer.begin(), thrust::plus<FloatType>());
            const auto end = std::chrono::high_resolution_clock::now();
            const double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            thrust::copy(resultBuffer.begin(), resultBuffer.end(), result.begin());
            return std::make_pair(result,  duration);
        }
    };

    template class ImplThrust<float>;
    template class ImplThrust<double>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<float>>::benchmark)
    ->Name("VecAdd-Thrust-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();
BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<double>>::benchmark)
    ->Name("VecAdd-Thrust-Double")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e7)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}