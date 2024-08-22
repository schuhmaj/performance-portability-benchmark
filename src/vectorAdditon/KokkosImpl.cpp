#include "VectorAddition.h"
#include <benchmark/benchmark.h>
#include "Kokkos_Core.hpp"


template<typename FloatType>
std::vector<FloatType> VectorAddition<FloatType>::operator()() {
    // Step 1: Define Kokkos Views
    Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostA{_inA.data(), _inA.size()};
    Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostB{_inB.data(), _inB.size()};
    Kokkos::View<FloatType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostC{_outC.data(), _outC.size()};

    // Step 2: Create managed views on the device side
    Kokkos::View<FloatType*> deviceA("deviceA", _inA.size());
    Kokkos::View<FloatType*> deviceB("deviceB", _inB.size());
    Kokkos::View<FloatType*> deviceC("deviceC", _outC.size());

    // Step 3: Copy data from host unmanaged views to device managed views
    Kokkos::deep_copy(deviceA, hostA);
    Kokkos::deep_copy(deviceB, hostB);

    // perform vector addition on device
    Kokkos::parallel_for("VecAdd", deviceA.size(), KOKKOS_LAMBDA(const int i) {
       deviceC(i) = deviceA(i) + deviceB(i);
    });

    // deep copy result back to host
    Kokkos::deep_copy(hostC, deviceC);
    return _outC;
}


template std::vector<float> VectorAddition<float>::operator()();
template std::vector<double> VectorAddition<double>::operator()();


int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard{argc, argv};
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}