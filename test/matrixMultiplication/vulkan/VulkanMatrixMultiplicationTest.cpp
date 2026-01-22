#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/vulkan/Impl_Vulkan.h"

TEST_P(MatrixMultiplicationTest, ImplVulkan_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplVulkan<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize,MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));