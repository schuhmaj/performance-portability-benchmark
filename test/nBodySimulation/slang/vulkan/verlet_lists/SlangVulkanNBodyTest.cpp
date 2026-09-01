#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/vulkan/verlet_lists/Impl_Vulkan.h"

TEST_P(NBodyTest, ImplVulkan_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplVulkan<float>>(size, 5);
}

TEST_F(NBodyEnergyTest, ImplVulkan_EnergyConservation) {
    // The Verlet-list implementations truncate the interaction at ParticleSimulationConfig::influenceRadius (4.0),
    // while the test evaluates the full, untruncated Lennard-Jones potential. Pairs crossing that radius
    // therefore produce a small, physical step in the measured potential energy (~7e-4 relative), which is not
    // an integration error. Hence the looser tolerance.
    this->runEnergyConservationTest<ppb::ImplVulkan<float>>(5e-3);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));