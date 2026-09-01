#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/kokkos/Impl_Kokkos.h"
#include "nBodySimulation/kokkos/Impl_KokkosReduction.h"

TEST_P(NBodyTest, ImplKokkos_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplKokkos<float>>(size);
}

TEST_P(NBodyTest, ImplKokkosReduction_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplKokkosReduction<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplKokkos_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplKokkos<float>>();
}

TEST_F(NBodyEnergyTest, ImplKokkosReduction_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplKokkosReduction<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100, 1000));