#include "Impl_Kokkos.h"
#include "Kokkos_Core.hpp"

namespace ppb {
    template <typename FloatType>
    std::vector<FloatType> ImplKokkos<FloatType>::operator()(const std::vector<FloatType> &a,
                                                               const std::vector<FloatType> &b, const MatrixMultiplicationConfig &config) {
        std::vector<FloatType> result(config.m * config.n, 0.0);

        Kokkos::View<FloatType**, Kokkos::LayoutLeft, Kokkos::SharedSpace, Kokkos::MemoryUnmanaged> hostA(const_cast<FloatType *>(a.data()), config.m, config.k);
        // Kokkos::View<FloatType**, Kokkos::LayoutLeft> devB("devB", config.k, config.n);
        // Kokkos::View<FloatType**, Kokkos::LayoutLeft> devC("devC", config.n, config.m);

        // auto hostA = Kokkos::create_mirror_view(devA);
        // std::memcpy(hostA.data(), a.data(), sizeof(FloatType) * config.m * config.k);
        // Kokkos::deep_copy(devA, hostA);
        // auto hostB = Kokkos::create_mirror_view(devB);
        // std::memcpy(hostB.data(), b.data(), sizeof(FloatType) * config.k * config.n);
        // Kokkos::deep_copy(devB, hostB);
        // auto hostC = Kokkos::create_mirror_view(devC);


        std::array<FloatType, 5> x{hostA(0,0), hostA(0,1), hostA(0,2), hostA(1,0), hostA(1,1)};


        return result;
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkos<float>;
    template class ImplKokkos<double>;
}