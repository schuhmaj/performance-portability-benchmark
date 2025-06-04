#include <benchmark/benchmark.h>
#include <iostream>
#include "Kokkos_Core.hpp"
#include "VectorAddition.h"

template <typename FloatType>
struct ppb::VectorAddition<FloatType>::impl {
    Kokkos::View<FloatType *> deviceA;
    Kokkos::View<FloatType *> deviceB;

    explicit impl(const std::vector<FloatType> &a, const std::vector<FloatType> &b) : deviceA{"deviceA", a.size()}, deviceB {"deviceB", b.size()} {
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
    _impl = std::make_unique<impl>(_inA, _inB);
}

template <typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    const size_t size = _impl->deviceA.size();
    Kokkos::View<FloatType *> result("result",size);
    const auto& deviceA = _impl->deviceA;
    const auto& deviceB = _impl->deviceB;

    Kokkos::parallel_for("VecAdd", size, KOKKOS_LAMBDA(const int i) {
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
