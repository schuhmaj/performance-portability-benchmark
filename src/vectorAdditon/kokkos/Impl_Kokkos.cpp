#include <benchmark/benchmark.h>
#include <iostream>
#include <utility>
#include "Kokkos_Core.hpp"
#include "vectorAdditon/VectorAddition.h"

namespace ppb {
    template <typename FloatType>
    struct KokkosImpl {
        using float_type = FloatType;
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            Kokkos::View<FloatType *> deviceA{"deviceA", size};
            Kokkos::View<FloatType *> deviceB{"deviceB", size};

            auto hostA = Kokkos::create_mirror_view(deviceA);
            std::copy(a.begin(), a.end(), hostA.data());
            Kokkos::deep_copy(deviceA, hostA);

            auto hostB = Kokkos::create_mirror_view(deviceB);
            std::copy(b.begin(), b.end(), hostB.data());
            Kokkos::deep_copy(deviceB, hostB);

            Kokkos::fence();
            Kokkos::Timer timer;
            Kokkos::View<FloatType *> result("result", size);
            Kokkos::parallel_for("VecAdd", size, KOKKOS_LAMBDA(const int i) {
                result(i) = deviceA(i) + deviceB(i);
            });
            Kokkos::fence();
            const double seconds = timer.seconds();

            const auto res_host = Kokkos::create_mirror_view(result);
            Kokkos::deep_copy(res_host, result);
            return std::make_pair(std::vector<FloatType>(res_host.data(), res_host.data() + res_host.size()), seconds * 1e9);
        }
    };

    template class KokkosImpl<float>;
    template class KokkosImpl<double>;
};

BENCHMARK(ppb::VectorAddition<ppb::KokkosImpl<ppb::VectorAdditionBenchmarkConf::float_type>>::benchmark)
    ->Name("VecAdd")
    ->RangeMultiplier(10)
    ->Range(ppb::VectorAdditionBenchmarkConf::MIN_SIZE, ppb::VectorAdditionBenchmarkConf::MAX_SIZE)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

int main(int argc, char **argv) {
    ppb::VectorAdditionBenchmarkConf::addContext("Kokkos");
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    Kokkos::ScopeGuard guard{argc, argv};
    // std::cout << "Default Execution Space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
