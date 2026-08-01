#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/slang/Impl_SlangVulkan.h"

TEST_P(MatrixMultiplicationTest, SlangVulkanImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplSlangVulkan<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));
