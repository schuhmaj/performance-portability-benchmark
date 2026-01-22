#include "vectorAdditon/cuda/Implementations.cuh"

#include <benchmark/benchmark.h>
#include <chrono>
#include <utility>
#include <vector>
#include "vectorAdditon/VectorAddition.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/host_vector.h"
#include "thrust/transform.h"

namespace ppb {
    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplThrust<FloatType>::operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
        const size_t size = a.size();
        thrust::device_vector<FloatType> deviceA(a.begin(), a.end());
        thrust::device_vector<FloatType> deviceB(b.begin(), b.end());
        std::vector<FloatType> result(size);
        thrust::device_vector<FloatType> resultBuffer(size);

        const auto start = std::chrono::high_resolution_clock::now();
        thrust::transform(thrust::device, deviceA.begin(), deviceA.end(),deviceB.begin(), resultBuffer.begin(), thrust::plus<FloatType>());
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_nanoseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        thrust::copy(resultBuffer.begin(), resultBuffer.end(), result.begin());
        return std::make_pair(result,  elapsed_nanoseconds);
    };

    template class ImplThrust<float>;
    template class ImplThrust<double>;
}
