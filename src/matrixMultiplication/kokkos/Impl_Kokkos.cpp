#include "Impl_Kokkos.h"
#include "Kokkos_Core.hpp"

namespace ppb {
    template <typename FloatType>
    std::vector<FloatType> ImplKokkos<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.m * config.n, 0.0);

        Kokkos::View<FloatType**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostA(const_cast<FloatType *>(a.data()), config.m, config.k);
        Kokkos::View<FloatType**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hostB(const_cast<FloatType *>(b.data()), config.k, config.n);
        Kokkos::View<FloatType**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> devC("devC", config.m, config.n);

        auto devA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace{}, hostA);
        auto devB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace{}, hostB);

        // Indexing this way: (row, column)
        Kokkos::parallel_for("matrix_multiplication", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {config.m, config.n}), KOKKOS_LAMBDA(const int i, const int j) {
            FloatType sum = 0.0;
            for (int entry = 0; entry < config.k; ++entry) {
                sum += devA(i, entry) * devB(entry, j);
            }
            devC(i, j) += sum;
        });
        typename Kokkos::View<FloatType**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>::HostMirror hostC = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, devC);
        std::copy(hostC.data(), hostC.data() + config.m * config.n, result.begin());
        return result;
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkos<float>;
    template class ImplKokkos<double>;
}