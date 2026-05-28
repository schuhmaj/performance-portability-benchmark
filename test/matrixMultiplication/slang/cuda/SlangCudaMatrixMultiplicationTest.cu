#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/slang/Impl_SlangCuda.cuh"

TEST_P(MatrixMultiplicationTest, SlangCudaImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplSlangCuda<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));
