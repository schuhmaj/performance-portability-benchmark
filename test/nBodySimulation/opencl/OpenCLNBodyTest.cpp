#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/opencl/Impl_OpenCL.h"

TEST_P(NBodyTest, ImplOpenCL_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplOpenCL<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplOpenCL_EnergyConservation) {
    this->runEnergyConservationTest<ppb::ImplOpenCL<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));