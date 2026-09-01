#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/alpaka/Impl_Alpaka.h"

TEST_P(NBodyTest, ImplAlpaka_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplAlpaka<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplAlpaka_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplAlpaka<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100, 1000));