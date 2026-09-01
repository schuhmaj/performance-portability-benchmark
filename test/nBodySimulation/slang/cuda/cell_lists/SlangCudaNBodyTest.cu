#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/slang/cell_lists/Impl_Slang_Cuda.cuh"

TEST_P(NBodyTest, ImplSlangCuda_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplSlangCuda<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplSlangCuda_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplSlangCuda<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));