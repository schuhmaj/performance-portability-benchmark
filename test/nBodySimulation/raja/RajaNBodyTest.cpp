#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/raja/Impl_Raja.h"

TEST_P(NBodyTest, ImplRaja_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplRaja<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplRaja_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplRaja<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100, 1000));
