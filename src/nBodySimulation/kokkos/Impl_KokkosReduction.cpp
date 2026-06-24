#include "Impl_KokkosReduction.h"

namespace ppb {

    template<typename FloatType>
    void ImplKokkosReduction<FloatType>::computeForces() {
        const size_t size = this->_particles->size();
        auto &force = this->_particles->forces;
        auto &position = this->_particles->positions;

        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;

        // One team per particle i; the team cooperatively reduces the force
        // contributions from all other particles j.
        TeamPolicy policy(size, Kokkos::AUTO);
        const Kokkos::Timer timer;
        Kokkos::parallel_for("compute_forces", policy, KOKKOS_LAMBDA(const typename TeamPolicy::member_type &team) {
            const int i = team.league_rank();

            constexpr FloatType sigmaSrc = 1.0;
            constexpr FloatType epsilonSrc = 1.0;
            constexpr FloatType sigma = (sigmaSrc + sigmaSrc) * 0.5;
            constexpr FloatType sigmaSquared = sigma * sigma;
            const FloatType epsilon24 = Kokkos::sqrt(epsilonSrc * epsilonSrc) * 24.0;

            FloatType sum0 = 0;
            FloatType sum1 = 0;
            FloatType sum2 = 0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, 0, size),
                [&](const int j, FloatType &s0, FloatType &s1, FloatType &s2) {
                    if (j == i) {
                        return;
                    }
                    FloatType dr[3];
                    FloatType dr2 = 0;
                    for (int k = 0; k < 3; ++k) {
                        dr[k] = position(i, k) - position(j, k);
                        dr2 += dr[k] * dr[k];
                    }

                    const FloatType invdr2 = 1.0 / dr2;
                    FloatType lj6 = sigmaSquared * invdr2;
                    lj6 = lj6 * lj6 * lj6;
                    const FloatType lj12 = lj6 * lj6;
                    const FloatType lj12m6 = lj12 - lj6;
                    const FloatType fac = epsilon24 * (lj12 + lj12m6) * invdr2;

                    s0 += dr[0] * fac;
                    s1 += dr[1] * fac;
                    s2 += dr[2] * fac;
                }, sum0, sum1, sum2);

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                force(i, 0) += sum0;
                force(i, 1) += sum1;
                force(i, 2) += sum2;
            });
        });
        Kokkos::fence();
        this->_timings.forceUpdateTime += (timer.seconds() * 1e9);
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkosReduction<float>;
    template class ImplKokkosReduction<double>;

};
