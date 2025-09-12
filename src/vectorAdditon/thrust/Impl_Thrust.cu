#include <benchmark/benchmark.h>
#include "VectorAddition.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/host_vector.h"
#include "thrust/transform.h"
namespace ppb {
    template <typename FloatType>
    struct ImplThrust {
        using float_type = FloatType;

        std::vector<FloatType> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            thrust::device_vector<FloatType> deviceA(a.begin(), a.end());
            thrust::device_vector<FloatType> deviceB(b.begin(), b.end());
            std::vector<FloatType> result(size);
            thrust::device_vector<FloatType> resultBuffer(size);

            thrust::transform(thrust::device, deviceA.begin(), deviceA.end(),deviceB.begin(), resultBuffer.begin(), thrust::plus<FloatType>());
            thrust::copy(resultBuffer.begin(), resultBuffer.end(), result.begin());
            return result;
        }
    };

    template class ImplThrust<float>;
    template class ImplThrust<double>;
}

BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<float>>::benchmark)->Name("VecAdd-Thrust-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();
BENCHMARK(ppb::VectorAddition<ppb::ImplThrust<double>>::benchmark)->Name("VecAdd-Thrust-Double")->RangeMultiplier(10)->Range(1e3, 1e7)->Complexity();

int main(int argc, char** argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}