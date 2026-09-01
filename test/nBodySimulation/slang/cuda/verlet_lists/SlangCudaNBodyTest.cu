#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/slang/verlet_lists/Impl_Slang_Cuda.cuh"

TEST_P(NBodyTest, ImplSlangCuda_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplSlangCuda<float>>(size);
}

TEST_F(NBodyEnergyTest, ImplSlangCuda_EnergyConservation) {
    // The Verlet-list implementations truncate the interaction at ParticleSimulationConfig::influenceRadius (4.0),
    // while the test evaluates the full, untruncated Lennard-Jones potential. Pairs crossing that radius
    // therefore produce a small, physical step in the measured potential energy (~7e-4 relative), which is not
    // an integration error. Hence the looser tolerance.
    this->runEnergyConservationTest<ppb::ImplSlangCuda<float>>(5e-3);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));