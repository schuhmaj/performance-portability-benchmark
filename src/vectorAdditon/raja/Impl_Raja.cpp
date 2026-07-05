#include <benchmark/benchmark.h>
#include <chrono>
#include <utility>
#include "RAJA/RAJA.hpp"
#include "vectorAdditon/VectorAddition.h"

namespace ppb {

#if defined(RAJA_ENABLE_CUDA)
    using ExecPolicy = RAJA::cuda_exec<256>;
    using Resource = RAJA::resources::Cuda;
#elif defined(RAJA_ENABLE_HIP)
    using ExecPolicy = RAJA::hip_exec<256>;
    using Resource = RAJA::resources::Hip;
#elif defined(RAJA_ENABLE_SYCL)
    using ExecPolicy = RAJA::sycl_exec<256>;
    using Resource = RAJA::resources::Sycl;
#elif defined(RAJA_ENABLE_OPENMP)
    using ExecPolicy = RAJA::omp_parallel_for_exec;
    using Resource = RAJA::resources::Host;
#else
    using ExecPolicy = RAJA::seq_exec;
    using Resource = RAJA::resources::Host;
#endif

    template <typename FloatType>
    struct RajaImpl {
        using float_type = FloatType;
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            Resource res = Resource::get_default();

            FloatType *deviceA = res.template allocate<FloatType>(size);
            FloatType *deviceB = res.template allocate<FloatType>(size);
            FloatType *deviceResult = res.template allocate<FloatType>(size);

            res.memcpy(deviceA, a.data(), size * sizeof(FloatType));
            res.memcpy(deviceB, b.data(), size * sizeof(FloatType));
            res.wait();

            const auto start = std::chrono::steady_clock::now();
            RAJA::forall<ExecPolicy>(res, RAJA::TypedRangeSegment<size_t>(0, size), [=] RAJA_HOST_DEVICE(const size_t i) {
                deviceResult[i] = deviceA[i] + deviceB[i];
            });
            res.wait();
            const auto end = std::chrono::steady_clock::now();
            const double nanoseconds = std::chrono::duration<double, std::nano>(end - start).count();

            std::vector<FloatType> result(size);
            res.memcpy(result.data(), deviceResult, size * sizeof(FloatType));
            res.wait();

            res.deallocate(deviceA);
            res.deallocate(deviceB);
            res.deallocate(deviceResult);
            return std::make_pair(std::move(result), nanoseconds);
        }
    };

    template class RajaImpl<float>;
    template class RajaImpl<double>;
};

BENCHMARK(ppb::VectorAddition<ppb::RajaImpl<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("RAJA");
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
