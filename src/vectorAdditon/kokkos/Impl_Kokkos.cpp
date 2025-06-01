#include <benchmark/benchmark.h>
#include <iostream>
#include "VectorAddition.h"
#include "Kokkos_Core.hpp"


template<typename FloatType>
std::vector<FloatType> ppb::VectorAddition<FloatType>::operator()() {
    // Step 1: Define Kokkos Views
    Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostA{_inA.data(), _inA.size()};
    Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostB{_inB.data(), _inB.size()};
    Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostC{_outC.data(), _outC.size()};

    // Step 2: Create managed views on the device side
    Kokkos::View<FloatType*> deviceA("deviceA", _inA.size());
    Kokkos::View<FloatType*> deviceB("deviceB", _inB.size());
    Kokkos::View<FloatType*> deviceC("deviceC", _outC.size());

    // Step 3: Copy data from host unmanaged views to device views
    Kokkos::deep_copy(deviceA, hostA);
    Kokkos::deep_copy(deviceB, hostB);

    // Step 4: Perform vector addition on device
    Kokkos::parallel_for("VecAdd", deviceA.size(), KOKKOS_LAMBDA(const int i) {
       deviceC(i) = deviceA(i) + deviceB(i);
    });

    // Step 5: Deep copy result back to host
    Kokkos::deep_copy(hostC, deviceC);
    checkValidity();
    return _outC;
}


// Instantiate a benchmark using single precision
template std::vector<float> ppb::VectorAddition<float>::operator()();
BENCHMARK(ppb::VectorAddition<float>::benchmark)->Name("VecAdd-Kokkos-Float")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();

// Instantiate a benchmark using double precision
template std::vector<double> ppb::VectorAddition<double>::operator()();
BENCHMARK(ppb::VectorAddition<double>::benchmark)->Name("VecAdd-Kokkos-Double")->RangeMultiplier(10)->Range(1e3, 1e8)->Complexity();


int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};

    std::cout << "Default Execution Space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}