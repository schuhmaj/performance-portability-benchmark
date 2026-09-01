#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/boost/Impl_Boost.h"

TEST_P(NBodyTest, ImplBoost_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplBoost<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplBoost_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplBoost<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));