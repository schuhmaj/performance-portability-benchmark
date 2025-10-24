#include <benchmark/benchmark.h>
#include <iostream>
#include <utility>
#include "Kokkos_Core.hpp"
#include "vectorAdditon/VectorAddition.h"

// This neat code using SharedMemory performs better for smaller vector sizes N
// However, the larger the vector becomes, the better a "pure" GPU version becomes

// template <typename FloatType>
// struct ppb::VectorAddition<FloatType>::impl {
//     Kokkos::View<FloatType *, Kokkos::SharedSpace> deviceA;
//     Kokkos::View<FloatType *, Kokkos::SharedSpace> deviceB;
//
//     explicit impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b) :
//         deviceA{"deviceA", size}, deviceB {"deviceB", size} {
//         std::copy(a.begin(), a.end(), deviceA.data());
//         std::copy(b.begin(), b.end(), deviceB.data());
//     }
// };

// Equivalent in runtime and the meaning compared to the formulation below
// The syntax is shorter

// template <typename FloatType>
// struct ppb::VectorAddition<FloatType>::impl {
//     Kokkos::View<FloatType *> deviceA;
//     Kokkos::View<FloatType *> deviceB;
//
//     explicit impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b) : deviceA{Kokkos::view_alloc("deviceA", Kokkos::WithoutInitializing)}, deviceB{Kokkos::view_alloc("deviceB", Kokkos::WithoutInitializing)} {
//         Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostA(const_cast<FloatType*>(a.data()), size);
//         Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostB(const_cast<FloatType*>(b.data()), size);
//         deviceA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace{}, hostA);
//         deviceB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace{}, hostB);
//     }
// };

namespace ppb {
    template <typename FloatType>
    struct KokkosImpl {
        using float_type = FloatType;
        std::pair<std::vector<FloatType>, double> operator()(const std::vector<FloatType> &a, const std::vector<FloatType> &b) {
            const size_t size = a.size();
            Kokkos::View<FloatType *> deviceA{"deviceA", size};
            Kokkos::View<FloatType *> deviceB{"deviceB", size};

            typename Kokkos::View<FloatType *>::HostMirror hostA = Kokkos::create_mirror_view(deviceA);
            std::copy(a.begin(), a.end(), hostA.data());
            Kokkos::deep_copy(deviceA, hostA);

            typename Kokkos::View<FloatType *>::HostMirror hostB = Kokkos::create_mirror_view(deviceB);
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

BENCHMARK(ppb::VectorAddition<ppb::KokkosImpl<float>>::benchmark)
    ->Name("VecAdd-Float-Kokkos")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
#ifdef PPB_MEASURE_ONLY_KERNEL
    ->UseManualTime()
#endif
    ->Complexity();

// BENCHMARK(ppb::VectorAddition<ppb::KokkosImpl<double>>::benchmark)
//     ->Name("VecAdd-Double-Kokkos")
//     ->RangeMultiplier(10)
//     ->Range(1e3, 1e8)
// #ifdef PPB_MEASURE_ONLY_KERNEL
//     ->UseManualTime()
// #endif
//     ->Complexity();


int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    Kokkos::ScopeGuard guard{argc, argv};
    // std::cout << "Default Execution Space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
