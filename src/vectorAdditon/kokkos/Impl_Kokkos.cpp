#include <benchmark/benchmark.h>
#include <iostream>
#include "Kokkos_Core.hpp"
#include "VectorAddition.h"

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

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {
    Kokkos::View<FloatType *> deviceA;
    Kokkos::View<FloatType *> deviceB;

    explicit impl(const size_t size, const std::vector<FloatType> &a, const std::vector<FloatType> &b) : deviceA{"deviceA", size}, deviceB {"deviceB", size} {
        typename Kokkos::View<FloatType *>::HostMirror hostA = Kokkos::create_mirror_view(deviceA);
        std::copy(a.begin(), a.end(), hostA.data());
        Kokkos::deep_copy(deviceA, hostA);
        typename Kokkos::View<FloatType *>::HostMirror hostB = Kokkos::create_mirror_view(deviceB);
        std::copy(b.begin(), b.end(), hostB.data());
        Kokkos::deep_copy(deviceB, hostB);
    }
};

template<typename FloatType>
void ppb::VectorAddition<FloatType>::init() {
    _impl = std::make_unique<impl>(_size, _inA, _inB);
}

template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    Kokkos::View<FloatType *> result("result",_size);
    const auto& deviceA = _impl->deviceA;
    const auto& deviceB = _impl->deviceB;

    Kokkos::parallel_for("VecAdd", _size, KOKKOS_LAMBDA(const int i) {
        result(i) = deviceA(i) + deviceB(i);
    });

    const auto res_host = Kokkos::create_mirror_view(result);
    Kokkos::deep_copy(res_host, result);
    return std::vector<FloatType>(res_host.data(), res_host.data() + res_host.size());
}


// Instantiate a benchmark using single precision
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)
    ->Name("VecAdd-Kokkos-Float")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();

// Instantiate a benchmark using double precision
template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)
    ->Name("VecAdd-Kokkos-Double")
    ->RangeMultiplier(10)
    ->Range(1e3, 1e8)
    ->Complexity();


int main(int argc, char **argv) {
    Kokkos::ScopeGuard guard{argc, argv};
    // std::cout << "Default Execution Space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
