#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/cuda/verlet_lists/include/Impl_Cuda.cuh"

TEST_P(NBodyTest, ImplCuda_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::cuda::nbody::ImplCuda<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplCuda_EnergyConservation) {
    this->runEnergyConservationTest<ppb::cuda::nbody::ImplCuda<float>>();
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100, 1000));