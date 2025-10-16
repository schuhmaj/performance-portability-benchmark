#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/kokkos/Impl_Kokkos.h"

TEST_P(NBodyTest, ImplKokkos_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplKokkos<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));