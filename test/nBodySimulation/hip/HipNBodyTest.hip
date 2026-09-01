#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/hip/Impl_Hip.h"

TEST_P(NBodyTest, ImplHip_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplHip<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplHip_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplHip<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));
