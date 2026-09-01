#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/stdpar/Impl_Stdpar.h"

TEST_P(NBodyTest, ImplStdpar_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplStdpar<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplStdpar_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplStdpar<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));
