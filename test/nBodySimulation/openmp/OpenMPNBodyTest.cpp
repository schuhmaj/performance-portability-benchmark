#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/openmp/Impl_OpenMP.h"

TEST_P(NBodyTest, ImplOpenMP_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplOpenMP<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplOpenMP_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplOpenMP<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));