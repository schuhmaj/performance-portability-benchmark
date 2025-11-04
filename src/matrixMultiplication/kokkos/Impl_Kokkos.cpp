#include "Impl_Kokkos.h"
#include "Kokkos_Core.hpp"

namespace ppb {
    template <typename FloatType>
    std::pair<std::vector<FloatType>, double> ImplKokkos<FloatType>::operator()(const std::vector<FloatType> &a,
                                                             const std::vector<FloatType> &b,
                                                             const MatrixMultiplicationConfig &config) {
        using ExecutionSpace = Kokkos::DefaultExecutionSpace;
        using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
        const int m = config.m;
        const int n = config.n;
        const int k = config.k;

        std::vector<FloatType> result(static_cast<size_t>(m) * n, static_cast<FloatType>(0));
        Kokkos::DefaultExecutionSpace exec;
        using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

        Kokkos::View<const FloatType**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostA(a.data(), m, k);
        Kokkos::View<const FloatType**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostB(b.data(), k, n);
        Kokkos::View<FloatType**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostC(result.data(), m, n);

        Kokkos::View<FloatType**, Kokkos::LayoutLeft, MemorySpace> devA("A", m, k);
        Kokkos::View<FloatType**, Kokkos::LayoutLeft, MemorySpace> devB("B", k, n);
        Kokkos::View<FloatType**, Kokkos::LayoutLeft, MemorySpace> devC("C", m, n);

        // Async transfers
        Kokkos::deep_copy(exec, devA, hostA);
        Kokkos::deep_copy(exec, devB, hostB);

        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace> policy(exec, {0, 0}, {m, n}, {32, 32});

        exec.fence();
        Kokkos::Timer timer;
        Kokkos::parallel_for("matrixMultiplication", policy, KOKKOS_LAMBDA(const int i, const int j) {
            FloatType sum = 0;
            for (int entry = 0; entry < k; ++entry) {
                sum += devA(i, entry) * devB(entry, j);
            }
            devC(i, j) = sum;
        });
        exec.fence();
        double seconds = timer.seconds();


        Kokkos::deep_copy(exec, hostC, devC);
        return std::make_pair(result, seconds * 1e9);
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkos<float>;
    template class ImplKokkos<double>;
}