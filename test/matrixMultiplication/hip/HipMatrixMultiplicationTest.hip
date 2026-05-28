#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/hip/Impl_HipNaive.h"

TEST_P(MatrixMultiplicationTest, HipNaiveImplementation_AllSizes) {
    const int size = GetParam();
    this->runTest<ppb::ImplHipNaive<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));
