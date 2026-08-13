#pragma once

#include <Kokkos_Core.hpp>
#include "nBodySimulation/kokkos/Impl_Kokkos.h"
#include "nBodySimulation/NBodySimulation.h"
#include "nBodySimulation/Particle.h"
#include "common/UtilityContainer.h"

namespace ppb {


    /**
     * @class ImplKokkos
     * Templated n-body simulation using Kokkos for parallelism, the Lennard-Jones potential, and velocity Verlet
     * integration.
     *
     * @tparam FloatType Floating-point type for simulation (e.g., float or double).
     */
    template <typename FloatType>
    class ImplKokkosReduction : public ImplKokkos<FloatType>{
    public:

        using float_type = FloatType;

        explicit ImplKokkosReduction(const ParticleSimulationConfig<FloatType> &config) : ImplKokkos<FloatType>(config) {}

        /**
         * Computes the inter-particle forces using the Lennard-Jones potential for all particles on the device,
         * accumulating the results in parallel.
         */
        void computeForces() final;
    };
} // namespace ppb
