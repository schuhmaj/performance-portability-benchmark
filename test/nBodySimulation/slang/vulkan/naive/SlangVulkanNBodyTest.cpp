#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "NBodyTest.h"
#include "nBodySimulation/slang/naive/Impl_Slang_Vulkan.h"

TEST_P(NBodyTest, ImplSlangVulkan_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplSlangVulkan<float>>(size, 5);
}

INSTANTIATE_TEST_SUITE_P(BySize, NBodyTest, ::testing::Values(10, 100));