#include "Impl_KokkosReduction.h"

namespace ppb {

    namespace {
        /**
         * 3-component vector value used as the reduction result when summing
         * the per-particle force contributions.
         */
        template <typename FloatType>
        struct ForceSum {
            FloatType data[3];
        };

        /**
         * Custom Kokkos reducer that sums ForceSum component-wise.
         *
         * A bare struct with only operator+= is NOT a valid reduction scalar for
         * a nested TeamThreadRange parallel_reduce on the CUDA backend: Kokkos
         * cannot form the cross-thread join, so the reduction silently returns
         * the identity (zero). Providing the full reducer interface (join/init)
         * makes the team reduction correct on both host and device.
         */
        template <typename FloatType, typename Space>
        struct ForceSumReducer {
            using reducer = ForceSumReducer;
            using value_type = ForceSum<FloatType>;
            using result_view_type = Kokkos::View<value_type, Space, Kokkos::MemoryUnmanaged>;

            KOKKOS_INLINE_FUNCTION explicit ForceSumReducer(value_type &value) : m_value(value) {}

            KOKKOS_INLINE_FUNCTION void join(value_type &dst, const value_type &src) const {
                dst.data[0] += src.data[0];
                dst.data[1] += src.data[1];
                dst.data[2] += src.data[2];
            }

            KOKKOS_INLINE_FUNCTION void init(value_type &val) const {
                val.data[0] = 0;
                val.data[1] = 0;
                val.data[2] = 0;
            }

            KOKKOS_INLINE_FUNCTION value_type &reference() const { return m_value; }
            KOKKOS_INLINE_FUNCTION result_view_type view() const { return result_view_type(&m_value); }
            KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }

        private:
            value_type &m_value;
        };
    } // namespace

    template<typename FloatType>
    void ImplKokkosReduction<FloatType>::computeForces() {
        const size_t size = this->_particles->size();
        auto &force = this->_particles->forces;
        auto &position = this->_particles->positions;

        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
        using Reducer = ForceSumReducer<FloatType, typename ExecSpace::memory_space>;

        // One team per particle i; the team cooperatively reduces the force
        // contributions from all other particles j into team_sum.
        TeamPolicy policy(size, Kokkos::AUTO);
        const Kokkos::Timer timer;
        Kokkos::parallel_for("compute_forces", policy, KOKKOS_LAMBDA(const typename TeamPolicy::member_type &team) {
            const int i = team.league_rank();

            constexpr FloatType sigmaSrc = 1.0;
            constexpr FloatType epsilonSrc = 1.0;
            constexpr FloatType sigma = (sigmaSrc + sigmaSrc) * 0.5;
            constexpr FloatType sigmaSquared = sigma * sigma;
            const FloatType epsilon24 = Kokkos::sqrt(epsilonSrc * epsilonSrc) * 24.0;

            ForceSum<FloatType> team_sum{};
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, 0, size),
                [&](const int j, ForceSum<FloatType> &sum) {
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

                    for (int k = 0; k < 3; ++k) {
                        sum.data[k] += dr[k] * fac;
                    }
                }, Reducer(team_sum));

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                force(i, 0) += team_sum.data[0];
                force(i, 1) += team_sum.data[1];
                force(i, 2) += team_sum.data[2];
            });
        });
        Kokkos::fence();
        this->_timings.forceUpdateTime += (timer.seconds() * 1e9);
    }

    /* Explicit Instantiation for float and double */
    template class ImplKokkosReduction<float>;
    template class ImplKokkosReduction<double>;

};
