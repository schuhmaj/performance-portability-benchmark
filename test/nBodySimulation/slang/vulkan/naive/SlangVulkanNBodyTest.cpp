#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/vulkan/naive/Impl_Vulkan.h"

TEST_P(NBodyTest, ImplVulkan_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplVulkan<float>>(size, 5);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));