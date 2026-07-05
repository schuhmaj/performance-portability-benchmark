#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MatrixMultiplicationTest.h"
#include "matrixMultiplication/raja/Impl_Raja.h"


TEST_P(MatrixMultiplicationTest, ImplRaja_Implementation) {
    const int size = GetParam();
    this->runTest<ppb::ImplRaja<float>>(size);
}

INSTANTIATE_TEST_SUITE_P(BySize, MatrixMultiplicationTest, ::testing::Values(2, 10, 32, 50, 64, 512));
